"""Milestone M3a recomputations: the multi-level optical-Bloch scattering-rate object (PLAN.md Sections 4.2.8, 8.1, 9.3).

Prints the [recomputed here] numbers behind the M3a ledger records:
  A. the 171Yb+ four-level detection rate against (Gamma/18) s_o/[1 + (2/9) s_o]: the maximum over the destabilizing
     field at fixed s_o, and the field it sits at;
  B. the leakage prefactors R_d, R_b from the angular algebra against Noek's and Crain's forms and the 3/49 ratio;
  C. the level-C sideband floor (Gamma/2 nu)^2 [alpha/cos^2 theta_L + 1/4] for sigma+ light along B on a mode along
     B and at 45 degrees, with recoil off (alpha -> 0 gives (Gamma/4 nu)^2), and the saturation error of the
     W(Delta -+ nu) closed form at Omega = 0.5 Gamma;
  D. the mixed pi + sigma emission channels: per-channel D against a single mean alpha;
  E. the Doppler regime nbar_D = (Gamma/4 nu)(1 + alpha) at Delta = -Gamma/2 from level A and level C;
  F. Morigi 2000 Fig. 3 from the Lambda level-C solve.

Run with  uv run python validation/scripts/check_bloch.py   (needs the package; about a minute).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions  # noqa: E402
from qutip_trap.light.beams import Beam  # noqa: E402
from qutip_trap.light.bloch import (  # noqa: E402
    BlochModel,
    beam_for_transition,
    carrier_weight,
    rate_coefficients,
    rate_coefficients_from_spectrum,
)
from qutip_trap.light.recoil import angular_factor  # noqa: E402
from qutip_trap.prep.level_c import level_c_relaxation_rate, level_c_steady_state  # noqa: E402
from qutip_trap.species import species  # noqa: E402
from qutip_trap.species.polarization import linear_polarization  # noqa: E402
from qutip_trap.species.raman import AtomicStructure  # noqa: E402
from qutip_trap.units import C_M_PER_S, HBAR_J_S, TWO_PI  # noqa: E402
from tests.bloch_fixtures import (  # noqa: E402
    LAMBDA_EXCITED,
    LAMBDA_GROUND_MINUS,
    LAMBDA_GROUND_PLUS,
    MASS_KG,
    TWO_LEVEL_EXCITED,
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    WAVELENGTH_M,
    circular_beam,
    gamma_rad_s,
    lambda_atom,
    sigma_plus_beam,
    structure,
    two_level_atom,
)


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


WAIST = 20e-6
yb = species("171Yb+")
line = yb.transition("S1/2-P1/2")
G_S = line.partial_rate_rad_s
D_HFP = TWO_PI * 2.105e9
D_HFS = TWO_PI * 12_642_812_118.5
BRIGHT = [f"S1/2 F=1 mF={m}" for m in (-1, 0, 1)]
DARK = ["S1/2 F=0 mF=0"]
MAGIC = tuple(linear_polarization((1.0, 0.0, 0.0), math.acos(1.0 / math.sqrt(3.0)), (0.0, 0.0, 1.0)))


def detection_model(s0: float, b_gauss: float, delta: float = 0.0) -> BlochModel:
    st = AtomicStructure(yb, b_gauss, (0.0, 0.0, 1.0))
    power = s0 * line.i_sat_w_m2 * math.pi * WAIST**2 / 2.0
    beam = beam_for_transition(
        st, "S1/2 F=1 mF=0", "P1/2 F=0 mF=0", delta, (1.0, 0.0, 0.0), MAGIC, power_w=power, waist_m=WAIST
    )
    return BlochModel(st, [beam], levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize"))


head(
    "A. 171Yb+ detection rate: exact four-level Bloch steady state against (Gamma/18) s_o/[1 + (2/9) s_o] (Section 8.1)"
)
print(
    f"Gamma_S/2pi = {G_S / TWO_PI / 1e6:.4f} MHz (partial), I_sat = {line.i_sat_w_m2 / 10:.3f} mW/cm^2, magic angle {math.degrees(math.acos(1 / math.sqrt(3))):.4f} deg"
)
for s0 in (0.01, 0.1, 1.0, 2.45):
    closed = (G_S / 18.0) * s0 / (1.0 + (2.0 / 9.0) * s0)
    best_ratio, best_b = 0.0, 0.0
    for b in np.geomspace(0.02, 40.0, 60):
        dr = detection_model(s0, b).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
        r = dr.R_bright_per_s / closed
        if r > best_ratio:
            best_ratio, best_b = r, b
    at_1g = detection_model(s0, 1.0).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2").R_bright_per_s / closed
    print(
        f"s_o = {s0:5.2f}: closed form {closed:.6e} s^-1; max over B of R_o/closed = {best_ratio:.5f} at B = {best_b:.3f} G "
        f"(Zeeman 1.4 MHz/G x B = {1.4 * best_b:.2f} MHz, pumping Gamma s_o/18 = {G_S * s0 / 18 / TWO_PI / 1e6:.4f} MHz); at 1 G: {at_1g:.5f}"
    )
print(
    "-> the closed form is the CEILING of the exact rate to 0.15-0.5 %, not reproduced 'to nine digits' (M3a finding)"
)

head(
    "B. Leakage prefactors from the angular algebra (Sections 8.1, 8.8): R_d, R_b at s_o = 0.1, B = 1 G, and Crain's s_o = 2.45"
)
for s0, b in ((0.1, 1.0), (2.45, 4.7)):
    dr = detection_model(s0, b).detection_rates(BRIGHT, DARK, line="S1/2<-P1/2")
    rd_noek = (2 / 3) * (1 / 3) * (G_S / 2) * (s0 / 3) * (G_S / (2 * D_HFP)) ** 2
    rd_crain = (1 / 3) * (G_S / 2) * (s0 / 3) * (G_S / (2 * D_HFP)) ** 2
    rb_noek = (2 / 3) * (G_S / 2) * (s0 / 3) * (G_S / (2 * (D_HFP + D_HFS))) ** 2
    print(
        # the slow-manifold rates come from the Liouvillian's eigen-decomposition, whose last digit depends on the BLAS
        # (Linux CI printed R_b = 0.60940 against 0.60939 here): four significant digits are what reproduces
        f"s_o = {s0}, B = {b} G: R_d = {dr.R_dark_pumping_per_s:.4g} Hz (Noek {rd_noek:.4f}, Crain {rd_crain:.4f}); "
        f"R_b = {dr.R_bright_pumping_per_s:.4g} Hz (Noek (2/3) form {rb_noek:.5f}); R_b/R_d = {dr.R_bright_pumping_per_s / dr.R_dark_pumping_per_s:.4f} (3/49 = {3 / 49:.5f}); "
        f"slow/fast separation {dr.separation:.3g}"
    )
print(
    "-> Noek's (2/3)(1/3) prefactor with s = s_o/3, i.e. R_d = (2/27)(Gamma/2) s_o (Gamma/2 Delta_HFP)^2; Crain's 1/3 is refuted"
)

head(
    "C. Level-C sideband floor: sigma+ along B, Delta = -nu, Gamma/nu = 0.02, Omega = 0.5 Gamma (two-level fixture)"
)
sp = two_level_atom()
st = structure(sp)
G = gamma_rad_s()
nu = 50.0 * G
for axis in ((0.0, 0.0, 1.0), (1 / math.sqrt(2), 0.0, 1 / math.sqrt(2))):
    cos_chi = float(np.dot(axis, (0, 0, 1)))
    alpha = angular_factor(1, cos_chi)
    mode = ModeSpec(nu, MASS_KG, axis, d=14, expected_n_max=3)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.5 * G, -nu)
    eta = mode.eta(beam.k_vector())
    cw = carrier_weight(alpha, beam.k_rad_per_m, float(np.dot(beam.k_vector(), axis)))
    print(
        f"mode axis {np.round(axis, 4)}: cos^2 theta_L = {cos_chi**2:.3f}, alpha_sigma = {alpha:.3f}, (eta~/eta)^2 = alpha/cos^2 = {cw:.3f}, eta = {eta:.5f}"
    )
    for recoil in ("off", "minimal", "marginal", "vector"):
        m = BlochModel(st, [beam], mode=mode, options=MultiLevelOptions(recoil=recoil))
        lc = level_c_steady_state(m.build)
        expected = (G / (2 * nu)) ** 2 * ((cw if recoil != "off" else 0.0) + 0.25)
        print(
            f"  recoil = {recoil:8s} ({len(m.build.c_ops):3d} operators): nbar_C = {lc.nbar:.6e}, closed form {expected:.6e}, ratio {lc.nbar / expected:.5f}"
        )
    m0 = BlochModel(st, [beam])
    sc = rate_coefficients_from_spectrum(m0, mode)
    rc = rate_coefficients(m0.w_of_offset(0), nu, cw)
    m_min = BlochModel(st, [beam], mode=mode, options=MultiLevelOptions(recoil="minimal"))
    w_c = level_c_relaxation_rate(m_min.build)
    print(
        f"  spectrum path: nbar = {sc.nbar:.6e}, W = {sc.cooling_rate_per_s:.5e} s^-1; level-C relaxation eigenvalue W = {w_c:.5e} s^-1 (ratio {w_c / sc.cooling_rate_per_s:.5f})"
    )
    print(
        f"  W(Delta -+ nu) closed form with the SATURATED W at Omega = 0.5 Gamma: nbar = {rc.nbar:.6e} = {rc.nbar / sc.nbar:.4f} x the exact value (1 + s = {1 + 2 * 0.25:.2f}); W = {rc.cooling_rate_per_s(eta):.5e}"
    )

head("D. Mixed pi + sigma emission channels: beam at 45 deg to B with pi and sigma components, mode along B")
k_hat = (1 / math.sqrt(2), 0.0, 1 / math.sqrt(2))
pol = (1 / math.sqrt(2) + 0j, 0j, -1 / math.sqrt(2) + 0j)
probe = Beam(WAVELENGTH_M, k_hat, pol, WAIST, 1e-3, (0.0, 0.0, 0.0))
om_pi = abs(st.single_photon_coupling_rad_s(st.state(TWO_LEVEL_GROUND), st.state(TWO_LEVEL_EXCITED), probe))
power = 1e-3 * (0.5 * G / om_pi) ** 2
omega_l = TWO_PI * (st.state(TWO_LEVEL_EXCITED).energy_hz - st.state(TWO_LEVEL_GROUND).energy_hz) - nu
beam45 = Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, WAIST, power, (0.0, 0.0, 0.0))
mode_z = ModeSpec(nu, MASS_KG, (0.0, 0.0, 1.0), d=14, expected_n_max=3)
m0 = BlochModel(st, [beam45])
ss = m0.steadystate()
print("steady-state populations:", {k: f"{v:.5e}" for k, v in ss.populations.items() if v > 1e-12})
sc = rate_coefficients_from_spectrum(m0, mode_z)
for recoil in ("minimal", "vector"):
    lc = level_c_steady_state(
        BlochModel(st, [beam45], mode=mode_z, options=MultiLevelOptions(recoil=recoil)).build
    )
    print(
        f"  level C ({recoil}): nbar = {lc.nbar:.6e}; spectrum path with per-channel alpha: {sc.nbar:.6e} (ratio {lc.nbar / sc.nbar:.5f})"
    )
kz = float(np.dot(beam45.k_vector(), (0, 0, 1)))
for a1 in (0.2, 0.4, 1 / 3):
    rc = rate_coefficients(m0.w_of_offset(0), nu, carrier_weight(a1, beam45.k_rad_per_m, kz))
    print(f"  single mean alpha = {a1:.4f}: nbar = {rc.nbar:.6e} ({rc.nbar / sc.nbar:.4f} x exact)")

head(
    "E. Doppler regime: nu = Gamma/20, Delta = -Gamma/2, sigma+ along B, mode along B: nbar_D = (Gamma/4 nu)(1 + alpha)"
)
nu_d = 0.05 * G
alpha = angular_factor(1, 1.0)
beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, -0.5 * G)
rc = rate_coefficients(
    BlochModel(st, [beam]).w_of_offset(0), nu_d, carrier_weight(alpha, beam.k_rad_per_m, beam.k_rad_per_m)
)
print(
    f"level A: nbar_D = {rc.nbar:.5f}; (Gamma/4 nu)(1 + alpha) = {(G / (4 * nu_d)) * (1 + alpha):.5f}; minus the 1/2 zero-point offset {(G / (4 * nu_d)) * (1 + alpha) - 0.5:.5f}"
)
mode_d = ModeSpec(nu_d, MASS_KG, (0.0, 0.0, 1.0), d=70, expected_n_max=40)
lc = level_c_steady_state(
    BlochModel(
        st,
        [beam],
        states=(TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS),
        mode=mode_d,
        options=MultiLevelOptions(recoil="minimal"),
    ).build
)
print(f"level C: nbar = {lc.nbar:.5f} (boundary population {lc.boundary_population:.1e})")
# the argmin is -(Gamma/2) sqrt(1 + s) only as nu/Gamma -> 0, and it sits 3e-3 Gamma below that at nu/Gamma = 0.05,
# so it is found with a bounded minimizer: the 0.025 Gamma grid this used to scan could only ever print -0.500


def _nbar_at(detuning_over_gamma: float, nu: float) -> float:
    return rate_coefficients(
        BlochModel(
            st,
            [
                sigma_plus_beam(
                    st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, detuning_over_gamma * G
                )
            ],
        ).w_of_offset(0),
        nu,
        carrier_weight(alpha, beam.k_rad_per_m, beam.k_rad_per_m),
    ).nbar


s_sat = 2.0 * 0.05**2
limit = -0.5 * math.sqrt(1.0 + s_sat)
print(f"-(1/2) sqrt(1 + s) at s = 2 (Omega/Gamma)^2 = {s_sat:.4f}: {limit:.6f} Gamma")
for ratio in (0.05, 0.01, 0.002):
    sol = minimize_scalar(
        lambda x, nu=ratio * G: _nbar_at(x, nu),
        bounds=(-1.0, -0.25),
        method="bounded",
        options={"xatol": 1e-7},
    )
    print(
        f"  argmin over Delta of nbar_D at nu/Gamma = {ratio:.3f} (bounded, xatol 1e-7): "
        f"Delta = {float(sol.x):.6f} Gamma (residual against the limit {float(sol.x) - limit:+.3e})"
    )

head(
    "F. Morigi 2000 Fig. 3 as printed: Omega_r = gamma, Omega_g = gamma/20, nu = gamma/10, eta = 0.145, Delta_g = Delta_r = 2.5 gamma"
)
lam = lambda_atom()
stl = structure(lam)
nu_m, om_r, om_g, delta_m = 0.1 * G, 1.0 * G, 0.05 * G, 2.5 * G
b_g = circular_beam(stl, LAMBDA_GROUND_MINUS, LAMBDA_EXCITED, om_g, delta_m, q=1, k_sign=1)
b_r = circular_beam(stl, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED, om_r, delta_m, q=-1, k_sign=-1)
x0_target = 0.145 / (2.0 * TWO_PI / WAVELENGTH_M)  # counter-propagating: eta = |k_g - k_r| x0 = 2 k x0
mass_m = HBAR_J_S / (2.0 * nu_m * x0_target**2)
mode_m = ModeSpec(nu_m, mass_m, (0.0, 0.0, 1.0), d=30, expected_n_max=8)


def eit_rate(sign: float) -> float:
    return (
        (om_g**2 / G)
        * G**2
        * nu_m**2
        / (G**2 * nu_m**2 + 4.0 * (om_r**2 / 4.0 - nu_m * (nu_m - sign * delta_m)) ** 2)
    )


a_plus, a_minus = eit_rate(+1.0), eit_rate(-1.0)
print(
    f"two-photon eta = {mode_m.eta(b_g.k_vector()) - mode_m.eta(b_r.k_vector()):.4f} (mass {mass_m / 1.66053906892e-27:.1f} u)"
)
print(
    f"Morigi Eq. (5): nbar_S = {a_plus / (a_minus - a_plus):.6f}; Eq. (6) rate eta^2 (A_- - A_+) = {0.145**2 * (a_minus - a_plus):.4e} s^-1; (gamma/4 Delta)^2 = 0.01"
)
for recoil in ("off", "minimal", "vector"):
    m = BlochModel(
        stl,
        [b_g, b_r],
        states=(LAMBDA_GROUND_MINUS, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED),
        mode=mode_m,
        options=MultiLevelOptions(leak="renormalize", recoil=recoil),
    )
    lc = level_c_steady_state(m.build)
    w = level_c_relaxation_rate(m.build)
    print(
        f"  level C (recoil {recoil:7s}): nbar = {lc.nbar:.6f}, P(0) = {lc.fock_populations[0]:.4f}, W = {w:.4e} s^-1"
    )
print(
    "-> the plan's 'Fig. 3' fixture (nu = 2.0068, Omega_1 = Omega_2 = 17, Delta = 70 MHz) is not this caption (M3a finding)"
)
print("done")
