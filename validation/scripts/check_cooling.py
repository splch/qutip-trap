"""Milestone M3 recomputations: laser cooling and state preparation (PLAN.md Sections 4.2.1-4.2.8, 9.3, 9.12, 9.13, 9.15, 9.17).

Prints the [recomputed here] numbers behind the M3 ledger records:
  A. recoil with participation: the axis energy identity for single and mixed crystals, the joint multi-mode kick's Cartesian
     sum rule, the Fock kernel moments and the free-atom recoil regressions;
  B. the Doppler stage: the rate framework against (Gamma/4 nu)(1 + alpha/cos^2) - 1/2 and the force model, the beam-detuning
     optimum, Monroe's 9Be+ triple from one beam and the identification of his theory value, 40Ca+ S-P-D against S-P;
  C. pulsed sideband cooling: Laguerre nodes, Che's accumulation centres and 75-pulse schedule, Rasmusson's fixed-pulse and
     multiorder schedules, the exactness of the sideband-ratio thermometry;
  D. EIT: the plan's Morigi fixture in the plan's sign, the RMP tuning, Roos's light shift, Lechner's rate ratio, and the
     Zeeman-resolved level-C solve of Roos's configuration;
  E. polarization-gradient cooling: the analytic limits and the Lindblad layer through the M3a builder;
  F. optical pumping: the 171Yb+ pump with its recoil heating and the Zeeman pump of a spin-zero ion.

Run with  uv run python validation/scripts/check_cooling.py   (needs the package; two to three minutes).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qutip_trap.api import Beam, HilbertSpace, ModeTruncation, RfDrive, Trap  # noqa: E402
from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions, decay_sum_rule_residual  # noqa: E402
from qutip_trap.light.bloch import (  # noqa: E402
    BlochModel,
    beam_for_transition,
)
from qutip_trap.light.recoil import (  # noqa: E402
    angular_factor,
    direction_quadrature,
    emission_lamb_dicke,
    free_recoil_energy_j,
    marginal_quadrature,
    multi_mode_kick,
    pattern_density,
    recoil_energy_ratio,
    recoil_kernel_matrix,
    recoil_projections,
    recoil_temperature_k,
    recoil_velocity_m_per_s,
)
from qutip_trap.prep.closed_forms import (  # noqa: E402
    doppler_force_nbar,
    doppler_limit_nbar,
    stenholm_coefficients,
)
from qutip_trap.prep.doppler import doppler_cooling, optimize_detuning, with_detuning_offset  # noqa: E402
from qutip_trap.prep.eit import (  # noqa: E402
    composed_coupling_rad_s,
    eit_cooling_rate_per_s,
    eit_nbar_at_rmp_tuning,
    eit_nbar_at_tuning,
    eit_rate_coefficients,
    eit_steady_state_nbar,
    lambda_level_c_model,
    light_shift_rad_s,
)
from qutip_trap.prep.level_c import level_c_relaxation_rate, level_c_steady_state  # noqa: E402
from qutip_trap.prep.polarization_gradient import (  # noqa: E402
    cooling_rate_per_s,
    fixed_phase_nbar,
    lin_perp_lin_pair,
    phase_averaged_minimum,
    polarization_gradient_model,
    saturation_for_xi,
    three_axis_lamb_dicke,
)
from qutip_trap.prep.pumping import optical_pumping  # noqa: E402
from qutip_trap.prep.sideband import (  # noqa: E402
    SidebandPulse,
    accumulation_centre,
    apply_pulses,
    laguerre_first_zero,
    mean_occupation,
    optimize_per_order,
    optimize_shared_duration,
    pi_time_s,
    sideband_ratio,
    stranded_index,
    stranded_population,
    thermal_distribution,
    thermal_ratio,
    truncated_tail,
)
from qutip_trap.species import species  # noqa: E402
from qutip_trap.species.polarization import linear_polarization, spherical_basis  # noqa: E402
from qutip_trap.species.raman import AtomicStructure  # noqa: E402
from qutip_trap.trap.crystal import solve_crystal  # noqa: E402
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, ELECTRON_MASS_U, HBAR_J_S, TWO_PI  # noqa: E402
from tests.atomic_fixtures import spin_zero_like  # noqa: E402
from tests.bloch_fixtures import (  # noqa: E402
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    power_for_rabi,
    structure,
    two_level_atom,
)


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def trap_for(freqs_hz, rf=None):  # type: ignore[no-untyped-def]
    return Trap(
        omega_hz=freqs_hz,
        axis_angle_rad=0.0,
        rf=rf,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def sigma_plus_along(st, k_hat, omega, delta, waist=20e-6):  # type: ignore[no-untyped-def]
    _m, _z, e_plus = spherical_basis(k_hat)
    pol = tuple(complex(x) for x in e_plus)
    power = power_for_rabi(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega, waist, pol, k_hat)
    omega_l = (
        TWO_PI * (st.state(TWO_LEVEL_EXCITED_PLUS).energy_hz - st.state(TWO_LEVEL_GROUND).energy_hz) + delta
    )
    return Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, waist, power, (0.0, 0.0, 0.0))


Z = (0.0, 0.0, 1.0)
OBLIQUE = (1.0 / math.sqrt(3.0),) * 3

# ---------------------------------------------------------------------------------------------------------------------------------
head(
    "A. Recoil with the per-ion participation (Section 4.2.8): energy identity, joint kick, kernel moments, unit regressions"
)
yb = species("171Yb+")
k369 = TWO_PI / 369.5e-9
chain = solve_crystal(trap_for((3.0e6, 2.9e6, 1.0e6)), (yb, yb))
for fam in ("axial", "transverse_1", "transverse_2"):
    ratios = [recoil_energy_ratio(chain, ion, k369, q, Z, fam) for ion in (0, 1) for q in (0, 1, None)]
    print(
        f"  two 171Yb+ ions, {fam:12s}: sum_m alpha c^2 (k x0)^2 hbar omega / [alpha_axis (hbar k)^2/(2 m)] - 1 = {max(abs(r - 1) for r in ratios):.1e} (both ions, pi/sigma/isotropic)"
    )
mixed = solve_crystal(trap_for((3.0e6, 2.9e6, 1.0e6), RfDrive(300.0, 60e6)), (yb, species("40Ca+")))
print(
    f"  171Yb+/40Ca+ pair (rf 60 MHz, axial modes {mixed.modes[0].omega_hz / 1e6:.3f} and {mixed.modes[1].omega_hz / 1e6:.3f} MHz, c_Yb = {mixed.modes[0].eigenvector[0]:.3f}): identity ratio - 1 = {recoil_energy_ratio(mixed, 0, k369, 1, Z, 'axial') - 1:.1e} (Yb), {recoil_energy_ratio(mixed, 1, k369, 1, Z, 'axial') - 1:.1e} (Ca) with each ion's own mass"
)
single = solve_crystal(trap_for((3.0e6, 2.9e6, 1.0e6)), (yb,))
space = HilbertSpace(
    ion_dims=(2,),
    resolved=tuple(ModeTruncation(m, 8, (0, 2), 0.2) for m in range(3)),
    enr_group=None,
    frozen=(),
)
quad = direction_quadrature(6, 8, axis=Z)
rho0 = qt.ket2dm(qt.tensor(qt.basis(2, 0), *[qt.basis(8, 0)] * 3))
out = 0.0 * rho0
for k_hat, w in zip(quad.directions, quad.weights):
    kick = multi_mode_kick(space, recoil_projections(single, 0, k369, tuple(k_hat)))
    out = out + w * float(pattern_density(1, float(k_hat[2]))) * kick * rho0 * kick.dag()
energy = sum(
    float(qt.expect(space.number(m), out)) * HBAR_J_S * single.modes[m].omega_rad_s for m in range(3)
)
quanta = [float(qt.expect(space.number(m), out)) for m in range(3)]
print(
    f"  joint kick of one sigma photon (6 x 8 directions): 1 - trace = {1.0 - float(out.tr()):.1e}, deposited energy / R - 1 = {energy / free_recoil_energy_j(k369, float(single.masses_kg[0])) - 1.0:.1e}"
)
for m, mode in enumerate(single.modes):
    alpha = angular_factor(1, float(np.dot(mode.e_hat, Z)))
    print(
        f"    mode {m} ({mode.family}, {mode.omega_hz / 1e6:.1f} MHz): <n> = {quanta[m]:.6e}, alpha eta_em^2 = {alpha * emission_lamb_dicke(single, 0, k369, m) ** 2:.6e}"
    )
kernel = recoil_kernel_matrix(60, 0.1, marginal_quadrature(1, 1.0))
n = np.arange(60)
print(
    f"  Fock kernel (eta_em = 0.1, sigma along B): 1 - column sums {[f'{1.0 - float(v):.1e}' for v in kernel.sum(axis=0)[:3]]}, mean kick of columns 0..3 minus alpha eta^2 = 0.004: {[f'{float(n @ kernel[:, c]) - c - 0.004:.1e}' for c in range(4)]}"
)
for name, mu, lam in (("133Cs 852 nm", 132.905, 852.347e-9), ("87Rb 780 nm", 86.909, 780.241e-9)):
    kk = TWO_PI / lam
    m = mu * ATOMIC_MASS_KG
    print(
        f"  {name}: v_r = {recoil_velocity_m_per_s(kk, m) * 1e3:.2f} mm/s, T_r = {recoil_temperature_k(kk, m) * 1e9:.1f} nK (k_B T_r = (hbar k)^2/m)"
    )
for name, mu, lam, g in (
    ("138Ba+ 493 nm", 137.905, 493.4e-9, 21e6),
    ("24Mg+ 280 nm", 23.985, 279.6e-9, 43e6),
):
    kk = TWO_PI / lam
    r = free_recoil_energy_j(kk, mu * ATOMIC_MASS_KG)
    print(
        f"  {name}: R/h = {r / (TWO_PI * HBAR_J_S) / 1e3:.1f} kHz, R/(hbar gamma) = {r / (HBAR_J_S * TWO_PI * g):.2e}"
    )

# ---------------------------------------------------------------------------------------------------------------------------------
head(
    "B. Doppler stage (Section 4.2.1): rate framework at any nu/Gamma, force model cross-check, detuning optimum, anchors"
)
sp = two_level_atom()
G = gamma_rad_s()
nu = 0.05 * G
crystal = solve_crystal(trap_for((2.5e6, 2.6e6, nu / TWO_PI)), (sp,))
st = structure(sp, b_hat=OBLIQUE)
beam = sigma_plus_along(st, OBLIQUE, 0.05 * G, -0.5 * G)
res = doppler_cooling(st, [beam], crystal)
print(
    f"two-level fixture, one sigma+ beam along B = (1,1,1)/sqrt3, Omega = Gamma/20, Delta = -Gamma/2 (W at rest {res.scattering_rate_per_s[0]:.4e} s^-1):"
)
for m in res.modes:
    alpha = angular_factor(1, float(np.dot(crystal.modes[m.mode].e_hat, OBLIQUE)))
    closed = doppler_limit_nbar(G, m.omega_rad_s, alpha / m.projection_cos2_max) - 0.5
    force = res.force_model_nbar.get(m.mode)
    print(
        f"  mode {m.mode} nu/Gamma = {m.omega_rad_s / G:.3f}: nbar = {m.nbar:.5f}, (Gamma/4 nu)(1 + alpha/cos^2) - 1/2 = {closed:.5f}, force model {'%.5f' % force if force is not None else 'not reported (nu/Gamma >= 0.1)'}, rate {m.rate_per_s:.4e} s^-1, eta sqrt(2 nbar + 1) = {m.lamb_dicke_thermal():.3f}"
    )
closed_form = doppler_cooling(st, [beam], crystal, method="closed_form")
print(
    f"  closed-form W(Delta -+ nu) method: nbar = {[round(m.nbar, 5) for m in closed_form.modes]} (spectrum path {[round(m.nbar, 5) for m in res.modes]})"
)


def evaluate(offset: float):  # type: ignore[no-untyped-def]
    return doppler_cooling(st, with_detuning_offset([beam], [0], offset), crystal, weights={0: 1.0})


best, opt = optimize_detuning(evaluate, (-0.8 * G, 0.4 * G), tolerance_rad_s=1e-3 * G)
print(
    f"  detuning optimum for the nu = Gamma/20 mode: Delta/Gamma = {(-0.5 * G + best) / G:.4f} (RMP -(1/2) sqrt(1 + s) = {-0.5 * math.sqrt(1 + 2 * 0.05**2):.4f}), nbar {opt.mode(0).nbar:.5f}"
)
both = doppler_cooling(
    st,
    [sigma_plus_along(st, OBLIQUE, 0.05 * G, -0.5 * G, waist=2e-3)],
    solve_crystal(trap_for((2.5e6, 2.6e6, nu / TWO_PI)), (sp, sp)),
)
one = doppler_cooling(
    st,
    [sigma_plus_along(st, OBLIQUE, 0.05 * G, -0.5 * G, waist=2e-3)],
    solve_crystal(trap_for((2.5e6, 2.6e6, nu / TWO_PI)), (sp, sp)),
    illuminated=[0],
)
print(
    f"  two-ion chain, COM axial: both ions illuminated nbar {both.mode(0).nbar:.5f} rate {both.mode(0).rate_per_s:.4e}; one ion nbar {one.mode(0).nbar:.5f} rate {one.mode(0).rate_per_s:.4e} (weight {one.mode(0).participation_weight:.3f})"
)

print(
    "Monroe 1995 9Be+: Gamma/2pi = 19.4 MHz, 313 nm, modes 11.2/18.2/29.8 MHz, Delta = -30 MHz; measured (0.47, 0.30, 0.18), theory 0.484 for x"
)
g_be = TWO_PI * 19.4e6
for alpha_name, alpha in (("sigma along B, 2/5", 0.4), ("isotropic 1/3", 1.0 / 3.0)):
    fm = doppler_force_nbar(g_be, -TWO_PI * 30e6, 0.0, TWO_PI * 11.2e6, alpha)
    rates = [
        stenholm_coefficients(0.01 * g_be, g_be, TWO_PI * f, -TWO_PI * 30e6, alpha).nbar
        for f in (11.2e6, 18.2e6, 29.8e6)
    ]
    print(
        f"  weight {alpha_name}: force model x-mode nbar {fm:.4f} (nu/Gamma = 0.58, outside its regime); rate framework, weak beam along each mode: {[round(r, 4) for r in rates]}"
    )
mass_be = (9.0121831 - ELECTRON_MASS_U) * ATOMIC_MASS_KG
be = two_level_atom(gamma_hz=19.4e6, wavelength_m=313e-9, mass_kg=mass_be)
cr_be = solve_crystal(trap_for((18.2e6, 29.8e6, 11.2e6)), (be,))
st_be = structure(be, b_hat=OBLIQUE)
for s in (0.1, 0.5, 2.0):
    r = doppler_cooling(
        st_be, [sigma_plus_along(st_be, OBLIQUE, math.sqrt(s / 2.0) * g_be, -TWO_PI * 30e6)], cr_be
    )
    by = {round(cr_be.modes[m.mode].omega_hz / 1e6, 1): round(m.nbar, 3) for m in r.modes}
    print(f"  one sigma+ beam along (1,1,1)/sqrt3 at s = {s}: nbar {by}")
print(
    "  -> the theory value 0.484 is the force model with alpha = 1/3; one oblique beam does not reproduce the triple (M3 finding)"
)

print(
    "40Ca+ S1/2-P1/2-D3/2 Doppler at -20 MHz (Roos 2000: measured nbar_z = 6.5(1.0) at 3.3 MHz, nbar_y = 16(2) at 1.6 MHz; two-level estimate 3 and 6)"
)
ca = species("40Ca+")
stc = AtomicStructure(ca, 4.0, Z)
l397, l866 = ca.transition("S1/2-P1/2"), ca.transition("D3/2-P1/2")
crc = solve_crystal(trap_for((1.7e6, 1.6e6, 3.3e6)), (ca,))
waist = 20e-6
pol397 = (1.0 / math.sqrt(2.0) + 0j, -1.0 / math.sqrt(2.0) + 0j, 0j)
k866 = (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
pol866 = (1.0 / math.sqrt(2.0) + 0j, 1.0 / math.sqrt(2.0) + 0j, 0j)


def ca_beams(s397: float, s866: float):  # type: ignore[no-untyped-def]
    p397 = s397 * l397.i_sat_w_m2 * math.pi * waist**2 / 2.0
    p866 = s866 * l866.i_sat_w_m2 * math.pi * waist**2 / 2.0
    b397 = beam_for_transition(
        stc, "S1/2 mJ=-1/2", "P1/2 mJ=-1/2", -TWO_PI * 20e6, OBLIQUE, pol397, power_w=p397, waist_m=waist
    )
    b866 = beam_for_transition(
        stc, "D3/2 mJ=-1/2", "P1/2 mJ=-1/2", 0.0, k866, pol866, power_w=p866, waist_m=waist
    )
    return [b397, b866]


z_mode, y_mode = crc.mode_index("axial", 0), crc.mode_index("transverse_2", 0)
for s397, s866 in ((0.3, 3.0), (1.0, 3.0), (3.0, 10.0)):
    full = doppler_cooling(stc, ca_beams(s397, s866), crc, levels=("S1/2", "P1/2", "D3/2"), cooling_beams=[0])
    two = doppler_cooling(
        stc,
        ca_beams(s397, s866)[:1],
        crc,
        levels=("S1/2", "P1/2"),
        options=MultiLevelOptions(leak="renormalize"),
    )
    print(
        f"  s_397 = {s397}, s_866 = {s866}: eight-state nbar_z = {full.mode(z_mode).nbar:.2f}, nbar_y = {full.mode(y_mode).nbar:.2f}; S-P two-level (D renormalized) {two.mode(z_mode).nbar:.2f}, {two.mode(y_mode).nbar:.2f}; W = {full.scattering_rate_per_s[0]:.3e}"
    )
print(
    "  -> the level structure changes nbar by < 10 % at these parameters; the measured excess is not reproduced as a level effect (M3 finding)"
)

# ---------------------------------------------------------------------------------------------------------------------------------
head(
    "C. Pulsed sideband cooling with exact matrix elements (Section 4.2.8; Che 2017; Rasmusson 2021) and thermometry (4.2.7)"
)
print(
    f"Laguerre first zeros (continuous degree): k = 1, eta = 0.3: {laguerre_first_zero(1, 0.3):.4f}; k = 2: {laguerre_first_zero(2, 0.3):.4f}; eta = 0.18: {laguerre_first_zero(1, 0.18):.4f}, {laguerre_first_zero(2, 0.18):.4f}"
)
print(
    f"stranded Fock states round(zero + k): {stranded_index(1, 0.3)}, {stranded_index(2, 0.3)}, {stranded_index(1, 0.18)} (Che's printed 40, 72 label the degree)"
)
om0 = TWO_PI * 150e3
p0 = thermal_distribution(17.0, 151)
print(
    f"Che 25Mg+: eta = 0.3, nbar = 17, 151 levels (tail {truncated_tail(17.0, 151):.2e}), 9 us pulses at Omega_0/2pi = 150 kHz"
)
for order, lo in ((1, 30), (2, 60)):
    line = []
    for count in (40, 150, 500):
        p = apply_pulses(p0, [SidebandPulse(order, 9e-6)] * count, 0.3, om0)
        c = accumulation_centre(p, order, 0.3)
        line.append(
            f"{count} pulses: peak n = {lo + int(np.argmax(p[lo:]))}, centre {c:.2f}, nbar {mean_occupation(p):.3f}"
        )
    print(f"  order {order}: " + "; ".join(line))
sched = (
    [SidebandPulse(2, 9e-6)] * 15
    + [SidebandPulse(1, 9e-6)] * 15
    + [SidebandPulse(2, 9e-6)] * 15
    + [SidebandPulse(1, 9e-6)] * 30
)
p = apply_pulses(p0, sched, 0.3, om0)
print(
    f"  Che's 75-pulse schedule (15 x k=2, 15 x k=1, 15 x k=2, 30 x k=1): nbar = {mean_occupation(p):.4f}, P(0) = {p[0]:.4f}, stranded above the k = 1 node {stranded_population(p, 1, 0.3):.2e}"
)
om0r = TWO_PI * 100e3
p14 = thermal_distribution(14.6, 200)
t_shared, nb_shared = optimize_shared_duration(p14, [1] * 25, 0.18, om0r, (1e-7, 200e-6))
p_pi = apply_pulses(p14, [SidebandPulse(1, pi_time_s(om0r, 0.18, 15, 1))] * 25, 0.18, om0r)
print(
    f"Rasmusson 171Yb+ (eta = 0.18): 25 first-order pulses from nbar = 14.6: shared optimum t = {t_shared * 1e6:.2f} us (Omega_0/2pi = 100 kHz) -> nbar {nb_shared:.3f}; pi time of n = 15 -> {mean_occupation(p_pi):.3f} (paper: nbar_sim = 3.57(58); measured 0.58 ratio, 8.0 SVD, 4.1 time-averaged)"
)
p15 = thermal_distribution(15.36, 220)
pulses, nb_order = optimize_per_order(p15, {3: 17, 2: 17, 1: 16}, 0.18, om0r, (1e-7, 200e-6))
durations = sorted({(pl.order, round(pl.duration_s * 1e6, 2)) for pl in pulses})
p_single = apply_pulses(p15, [SidebandPulse(1, t_shared)] * 50, 0.18, om0r)
print(
    f"  50 multiorder pulses (17 x k=3, 17 x k=2, 16 x k=1, one duration per order, higher first): nbar {nb_order:.4f}, durations (order, us) {durations} (paper: 0.06 with per-pulse optimization; independent durations reach 0.11-0.12 here)"
)
print(
    f"  50 first-order pulses: nbar {mean_occupation(p_single):.3f}; population at/above the node n = 113: {stranded_population(p_single, 1, 0.18):.2e} = thermal tail {(15.36 / 16.36) ** 113:.2e}, {float(np.dot(np.arange(113, 220), p_single[113:])):.3f} quanta (paper: 'about 0.3')"
)
pth = thermal_distribution(8.0, 400)
devs = [
    abs(sideband_ratio(pth, 1.5, om0r, k, t) - thermal_ratio(8.0, k))
    for k in (1, 2)
    for t in (1e-6, 3e-6, 7.7e-6)
]
print(
    f"Turchette sideband ratio, eta = 1.5, nbar = 8, k = 1, 2, three durations: max |R_k - (nbar/(nbar+1))^k| = {max(devs):.1e}"
)

# ---------------------------------------------------------------------------------------------------------------------------------
head("D. EIT cooling (Section 4.2.3): closed forms in the plan's sign and the Zeeman-resolved level C")
nu_m, gam_m, o1, o2, d_m = 2.0068, 20.0, 17.0, 17.0, 70.0
om_r = composed_coupling_rad_s([o1, o2])
# the plan's own fixture has Omega_g = 17 against gamma = 20 MHz: outside the weak-probe regime the closed form
# assumes, so the Section 4.2.8 (vii) escape is passed explicitly (the recorded M3a finding on this fixture)
rc = eit_rate_coefficients(o1, om_r, nu_m, d_m, gam_m, allow_saturation=True)
print(
    f"plan fixture (MHz): Omega_r = sqrt(17^2 + 17^2) = {om_r:.4f}, Delta = +70: <n>_S = {eit_steady_state_nbar(om_r, nu_m, d_m, gam_m):.6f} ((gamma/4 Delta)^2 = {eit_nbar_at_tuning(gam_m, d_m):.7f}); single 17: {eit_steady_state_nbar(17.0, nu_m, d_m, gam_m):.4f}"
)
print(
    f"  A_- = {rc.A_minus_per_s:.4f}, A_+ = {rc.A_plus_per_s:.5f} MHz (Section 4.2.3's formula: A_- = Omega_g^2/gamma at the tuning; the plan's 9.13 row 123 prints 7.225 and 0.036676, half); W = eta^2 (A_- - A_+) at eta = 0.02: {0.02**2 * rc.cooling_rate_bare_per_s * 1e3:.3f} kHz (row: 2.875)"
)
try:
    eit_steady_state_nbar(om_r, nu_m, -d_m, gam_m)
except Exception as exc:  # noqa: BLE001
    print(f"  Delta = -70 (the caption's sign read in the plan's convention): {type(exc).__name__}")
print(
    f"RMP tuning Omega_r^2 = 4 nu Delta at gamma = 1, nu = 0.3, Delta = 5: exact {eit_nbar_at_rmp_tuning(1.0, 0.3, 5.0):.8f} = closed form {eit_steady_state_nbar(math.sqrt(6.0), 0.3, 5.0, 1.0):.8f}; the (Delta + nu) form {(1 + 4 * 0.09) / (16 * 5 * 5.3):.8f}"
)
print(
    f"Roos 2000: Delta = 70 MHz, Omega_sigma = 21.4 MHz -> light shift {light_shift_rad_s(70.0, 21.4):.4f} MHz (his 1.6); at delta = nu = 1.6 MHz the closed form gives {eit_nbar_at_tuning(21.57, 70.0):.5f} against his measured 0.18"
)
gam_ca, om_s, om_p = 21.57, 30.0, 6.2
for shift in (2.2, 2.25, 2.3):
    d_l = brentq(lambda d, target=shift: light_shift_rad_s(d, om_s) - target, 10.0, 1000.0)
    r_hi = eit_cooling_rate_per_s(
        1.0 / math.sqrt(3.29), om_p, om_s, 3.29, d_l, gam_ca, allow_saturation=True
    )
    r_lo = eit_cooling_rate_per_s(
        1.0 / math.sqrt(1.13), om_p, om_s, 1.13, d_l, gam_ca, allow_saturation=True
    )
    print(
        f"Lechner 2016 (Omega_sigma = 30, Omega_pi = 6.2 MHz, light shift {shift} MHz -> Delta = {d_l:.2f} MHz): R(3.29)/R(1.13) with eta^2 ∝ 1/nu = {r_hi / r_lo:.3f} (measured 17/5 = 3.4; the plan's 9.12 row prints the inverse 2.9)"
    )
m40 = ca.mass_u * ATOMIC_MASS_KG
d_l = brentq(lambda d: light_shift_rad_s(d, om_s) - 2.25, 10.0, 1000.0)
for f_mode in (1.13, 3.29):
    bare = eit_rate_coefficients(
        TWO_PI * om_p * 1e6,
        TWO_PI * om_s * 1e6,
        TWO_PI * f_mode * 1e6,
        TWO_PI * d_l * 1e6,
        TWO_PI * gam_ca * 1e6,
        allow_saturation=True,
    ).cooling_rate_bare_per_s
    eta1 = (TWO_PI / 397e-9) * math.sqrt(HBAR_J_S / (2 * m40 * TWO_PI * f_mode * 1e6))
    print(
        f"  nu = {f_mode} MHz: A_- - A_+ = {bare:.4e} s^-1, single-beam eta = {eta1:.4f}; eta needed for the measured 19e3 s^-1: {math.sqrt(19e3 / bare):.4f} (|Delta k|/k = {math.sqrt(19e3 / bare) / eta1:.2f})"
    )
# level C: Roos configuration on 40Ca+ S1/2 + P1/2 (D3/2 renormalized): sigma+ along B = z and a pi beam along x; mode along Delta k
sigma_pol = (-1.0 / math.sqrt(2.0) + 0j, -1j / math.sqrt(2.0), 0j)
delta_roos = TWO_PI * 70e6
st_r = AtomicStructure(ca, 1e-6, Z)
probe_s = beam_for_transition(
    st_r, "S1/2 mJ=-1/2", "P1/2 mJ=1/2", delta_roos, Z, sigma_pol, power_w=1e-3, waist_m=waist
)
probe_p = beam_for_transition(
    st_r,
    "S1/2 mJ=1/2",
    "P1/2 mJ=1/2",
    delta_roos,
    (1.0, 0.0, 0.0),
    (0j, 0j, 1.0 + 0j),
    power_w=1e-3,
    waist_m=waist,
)
om_s0 = abs(st_r.single_photon_coupling_rad_s(st_r.state("S1/2 mJ=-1/2"), st_r.state("P1/2 mJ=1/2"), probe_s))
om_p0 = abs(st_r.single_photon_coupling_rad_s(st_r.state("S1/2 mJ=1/2"), st_r.state("P1/2 mJ=1/2"), probe_p))
nu_y = TWO_PI * 1.6e6
om_sigma = 2.0 * math.sqrt(
    nu_y * (nu_y + delta_roos)
)  # the coupling that puts delta exactly on the 1.6 MHz mode
b_sigma = beam_for_transition(
    st_r,
    "S1/2 mJ=-1/2",
    "P1/2 mJ=1/2",
    delta_roos,
    Z,
    sigma_pol,
    power_w=1e-3 * (om_sigma / om_s0) ** 2,
    waist_m=waist,
)
b_pi = beam_for_transition(
    st_r,
    "S1/2 mJ=1/2",
    "P1/2 mJ=1/2",
    delta_roos,
    (1.0, 0.0, 0.0),
    (0j, 0j, 1.0 + 0j),
    power_w=1e-3 * (TWO_PI * 3e6 / om_p0) ** 2,
    waist_m=waist,
)
axis = (-1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))  # along Delta k = k_sigma - k_pi
mode_r = ModeSpec(nu_y, m40, axis, d=24, expected_n_max=6)
lc_model = lambda_level_c_model(st_r, [b_sigma, b_pi], mode_r, levels=("S1/2", "P1/2"))
lc = level_c_steady_state(lc_model.build)
eta_two = mode_r.eta(b_sigma.k_vector()) - mode_r.eta(b_pi.k_vector())
w_c = level_c_relaxation_rate(lc_model.build)
closed = eit_steady_state_nbar(om_sigma, nu_y, delta_roos, TWO_PI * gam_ca * 1e6)
print(
    f"level C, Roos geometry on 40Ca+ S1/2 + P1/2 (sigma+ along B, pi beam along x, Omega_sigma/2pi = {om_sigma / TWO_PI / 1e6:.2f} MHz for delta = nu, Omega_pi/2pi = 3 MHz, mode along Delta k, eta = {eta_two:.4f}): nbar = {lc.nbar:.5f} (closed form {closed:.5f}, (gamma/4 Delta)^2 = {eit_nbar_at_tuning(gam_ca, 70.0):.5f}), P(0) = {lc.fock_populations[0]:.4f}, W = {w_c:.4e} s^-1 (closed form {eta_two**2 * eit_rate_coefficients(TWO_PI * 3e6, om_sigma, nu_y, delta_roos, TWO_PI * gam_ca * 1e6, allow_saturation=True).cooling_rate_bare_per_s:.4e}), boundary {lc.boundary_population:.1e}"
)

# ---------------------------------------------------------------------------------------------------------------------------------
head(
    "E. Polarization-gradient cooling (Section 4.2.4): analytic limits and the Lindblad layer through the M3a builder"
)
xi_min, n_min = phase_averaged_minimum()
print(
    f"fixed phase: <n_0> = xi + 1/(4 xi) - 1/2, minimum {fixed_phase_nbar(0.5):.10f} at xi = 0.5; phase-averaged minimum {n_min:.10f} at xi = {xi_min:.10f}"
)
print(
    f"Joshi operating point: s for xi = 1.35 at Delta = 2 pi x 210 MHz, omega_z = 2 pi x 1088 kHz: {saturation_for_xi(1.35, TWO_PI * 210e6, TWO_PI * 1088e3):.5f} (source 0.021(2))"
)
eta_ca = (TWO_PI / 397e-9) * math.sqrt(HBAR_J_S / (2 * m40 * TWO_PI * 1088e3))
w_j = cooling_rate_per_s(eta_ca, TWO_PI * 21.57e6, 0.02098, 1.35, 0.0)
print(
    f"  W(phi = 0) = {w_j:.3e} s^-1, phase-averaged {w_j / 2:.3e} (the plan quotes ~6.6e4 from the source); window W < delta = {TWO_PI * 60e3:.2e} < omega = {TWO_PI * 1088e3:.2e} rad/s"
)
etas = three_axis_lamb_dicke(TWO_PI / 369.5e-9, 170.9363 * ATOMIC_MASS_KG, (0.790e6, 0.766e6, 0.525e6))
print(
    f"171Yb+ three-axis triple (Ejtemaee): eta = ({etas[0]:.5f}, {etas[1]:.5f}, {etas[2]:.5f}) (source 0.052, 0.053, 0.090)"
)
st_pg = AtomicStructure(ca, 1e-6, Z)
line_ca = ca.transition("S1/2-P1/2")
for f_mode, delta_hz, xi_target in ((4.0e6, 500e6, 1.0), (4.0e6, 500e6, 0.5)):
    mode = ModeSpec(TWO_PI * f_mode, m40, Z, d=20, expected_n_max=6)
    lam = TWO_PI * C_M_PER_S / (TWO_PI * (line_ca.frequency_hz + delta_hz))
    probe = polarization_gradient_model(
        st_pg, lin_perp_lin_pair(lam, 1e-3, 20e-6, delta_hz), mode, lower="S1/2 mJ=-1/2", upper="P1/2 mJ=1/2"
    )
    om_target = math.sqrt(
        xi_target
        * 4.0
        * mode.omega_rad_s
        * (probe.delta_rad_s**2 + probe.gamma_rad_s**2 / 4.0)
        / probe.delta_rad_s
    )
    power = 1e-3 * (om_target / probe.omega_d1_sigma_rad_s) ** 2
    lcp = polarization_gradient_model(
        st_pg, lin_perp_lin_pair(lam, power, 20e-6, delta_hz), mode, lower="S1/2 mJ=-1/2", upper="P1/2 mJ=1/2"
    )
    b = lcp.model.build
    ss = level_c_steady_state(b)
    w_lc = level_c_relaxation_rate(b)
    s_j = 1.5 * lcp.omega_d1_sigma_rad_s**2 / 2.0 / (lcp.delta_rad_s**2 + lcp.gamma_rad_s**2 / 4.0)
    node = polarization_gradient_model(
        st_pg,
        lin_perp_lin_pair(lam, power, 20e-6, delta_hz, phase_rad=math.pi / 4),
        mode,
        lower="S1/2 mJ=-1/2",
        upper="P1/2 mJ=1/2",
    )
    n_node = level_c_steady_state(node.model.build).nbar
    print(
        f"Lindblad layer, 40Ca+ S1/2 + P1/2, omega = 2 pi x {f_mode / 1e6:.0f} MHz, Delta = +{delta_hz / 1e6:.0f} MHz, xi = {lcp.xi:.4f} (eta = {lcp.eta:.4f}, s_Joshi = {s_j:.4f}): {lcp.n_collapse_operators} collapse operators (3 q x 3 recoil classes), sum rule {decay_sum_rule_residual(b):.1e}"
    )
    print(
        f"  steady state nbar = {ss.nbar:.4f} vs analytic phi = 0 {fixed_phase_nbar(lcp.xi):.4f}; relaxation rate {w_lc:.4e} vs analytic W(0) {cooling_rate_per_s(lcp.eta, lcp.gamma_rad_s, s_j, lcp.xi, 0.0):.4e} s^-1; at the gradient node phi = pi/4 nbar = {n_node:.3f} (analytic: uncooled); boundary {ss.boundary_population:.1e}"
    )

# ---------------------------------------------------------------------------------------------------------------------------------
head(
    "F. Optical pumping (Section 4.2.6): 171Yb+ into |0,0> with the recoil heating of its photons; Zeeman pumping of a spin-zero ion"
)
st_yb = AtomicStructure(yb, 5.0, Z)
line_yb = yb.transition("S1/2-P1/2")
magic = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), Z))
pump = beam_for_transition(
    st_yb,
    "S1/2 F=1 mF=0",
    "P1/2 F=1 mF=0",
    0.0,
    (1.0, 0.0, 0.0),
    magic,
    power_w=0.5 * line_yb.i_sat_w_m2 * math.pi * waist**2 / 2.0,
    waist_m=waist,
)
model = BlochModel(st_yb, [pump], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))
resp = optical_pumping(model, ["S1/2 F=0 mF=0"], duration_s=30e-6, samples=6001, crystal=single, ion=0)
print(
    f"171Yb+ pump on F=1 -> F'=1 at s_o = 0.5, 5 G: photons {resp.photons_scattered:.4f}, error at 30 us {resp.preparation_error:.2e} (steady state {resp.steady_state_error:.2e}), 99.99 % at {resp.time_to_reach_s * 1e6:.2f} us"
)
for m, mode in enumerate(single.modes):
    print(
        f"  recoil heating of mode {m} ({mode.family}, {mode.omega_hz / 1e6:.1f} MHz): {resp.motional_heating_quanta[m]:.3e} quanta (eta_em = {emission_lamb_dicke(single, 0, k369, m):.4f})"
    )
sz = spin_zero_like()
st_z = AtomicStructure(sz, 1.0, Z)
lz = sz.transition("S1/2-P1/2")
bz = beam_for_transition(
    st_z,
    "S1/2 mJ=-1/2",
    "P1/2 mJ=1/2",
    0.0,
    Z,
    sigma_pol,
    power_w=0.05 * lz.i_sat_w_m2 * math.pi * waist**2 / 2.0,
    waist_m=waist,
)
resz = optical_pumping(
    BlochModel(st_z, [bz], levels=("S1/2", "P1/2")),
    ["S1/2 mJ=1/2"],
    duration_s=40e-6,
    samples=8001,
    initial=["S1/2 mJ=-1/2"],
)
print(
    f"spin-zero ion, sigma+ on S1/2 -> P1/2 from |-1/2>: photons {resz.photons_scattered:.5f} (pi branching 1/3 into the target: exactly 3), error {resz.preparation_error:.1e}"
)
print("done")
