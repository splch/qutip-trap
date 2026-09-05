"""Run 5 'constants' topic: named single-source items.

Recomputes every number tagged [recomputed here] in the Run 5 additions to Sections 8.1
(coherent population trapping and the saturation ceilings), 4.5.7 (metastable D-state
lifetimes, blackbody M1 mixing, collisional quenching, reshelving), 4.5.5 (Raman and
Rayleigh laser scattering, Wineland's detuning optimum) and 4.2.4 (polarization-gradient
cooling), plus the g_F and Breit-Rabi traps of Sections 4.5.6 and 13.

Run with
  uv run --python 3.13 --with qutip --with numpy --with scipy --with sympy --with mpmath python check_constants.py
"""

import numpy as np
from scipy.constants import (
    h,
    hbar,
    k as k_B,
    c,
    e,
    physical_constants,
    atomic_mass,
    pi,
)
from scipy.integrate import dblquad
from scipy.optimize import curve_fit, minimize_scalar
from sympy import S as Sym, sqrt as ssqrt
from sympy.physics.wigner import wigner_3j

mu_B_over_h = physical_constants["Bohr magneton in Hz/T"][0]  # Hz/T
a0_m = physical_constants["Bohr radius"][0]
t_au = physical_constants["atomic unit of time"][0]
m_e_over_m_p = physical_constants["electron-proton mass ratio"][0]

sep = lambda s: print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ----------------------------------------------------------------------------------
sep("1. Saturation ceilings of a closed Zeeman-degenerate manifold (Berkeland Sec. IV)")


# ----------------------------------------------------------------------------------
def ceiling(n_g, n_e):
    return n_e / (n_g + n_e)


for label, Ji, Jf in (("J_i=1 <-> J_f=0 (171Yb+ F=1 -> F'=0)", 1, 0),):
    n_g, n_e = 2 * Ji + 1, 2 * Jf + 1
    print(
        f"P_f^max[{label}] = (2J_f+1)/((2J_i+1)+(2J_f+1)) = {n_e}/{n_g + n_e} = "
        f"{ceiling(n_g, n_e):.10f}   -> R_gamma^max = Gamma/{(n_g + n_e) // n_e}"
    )
print(
    f"P_f^max[Lambda system, n_g=2, n_e=1]        = {ceiling(2, 1):.10f}   -> Gamma/3"
)
print(
    f"P_f^max[ideal two-level, n_g=1, n_e=1]      = {ceiling(1, 1):.10f}   -> Gamma/2"
)

# closed form Eq. (13)-(14) at the optima cos^2 theta = 1/3, delta_B = Omega/4, Delta = 0
Pf = lambda Om, gam=1.0: (Om**2 / 12) / ((gam / 2) ** 2 + Om**2 / 3)
for Om in (np.sqrt(3) / 5, 1.0, 100.0):
    print(f"Eq.(13) at optima, Omega = {Om:.6f} gamma: P_f = {Pf(Om):.6f}")
print(f"Eq.(13) ceiling  lim Omega->inf           : P_f = {1 / 4:.6f} (exact 1/4)")

# argmax over delta_B of the Eq. (14) width terms, and over theta_BE
Om = np.sqrt(3) / 5
w = lambda dB: Om**4 / (16 * dB**2) + 16 * dB**2
r = minimize_scalar(w, bracket=(1e-3, 1.0), method="brent")
print(
    f"argmin_delta_B [Omega^4/(16 dB^2) + 16 dB^2] = {r.x:.7f}  vs Omega/4 = {Om / 4:.7f}"
)
ang = lambda u: -u * (1 - u) / (1 + 3 * u)  # u = cos^2 theta_BE
r2 = minimize_scalar(ang, bounds=(0.0, 1.0), method="bounded")
print(
    f"argmax_theta_BE of cos^2 sin^2/(1+3cos^2): cos^2 = {r2.x:.7f} (1/3), "
    f"theta_BE = {np.degrees(np.arccos(np.sqrt(r2.x))):.4f} deg "
    f"(arctan sqrt2 = {np.degrees(np.arctan(np.sqrt(2))):.4f} deg)"
)
# the identity that makes the ceiling exactly 1/4
u = 1 / 3
X = Om**2 * u * (1 - u) / (1 + 3 * u)
print(
    f"min_dB(terms 2+3 of Eq.14) / (Eq.13 numerator coeff) = "
    f"{(3 * X) / ((3 / 4) * Om**2 * u * (1 - u) / (1 + 3 * u)):.10f}  (exactly 4)"
)

# Lambda-system rate equation, Eq. (21): 1/P_f = 3 + (1-a)g/R_if + a g/R_df
alpha_br = 1 / 13
for R in (1e0, 1e2, 1e4, 1e8):
    inv = 3 + (1 - alpha_br) / R + alpha_br / R
    print(f"Eq.(21) at R_if = R_df = {R:.0e} gamma: P_f = {1 / inv:.8f}")

# trace preservation of Eq. (12) with the correct q = m_f - m_i, and the wrong-sign variant
print("\nEq.(12) feeding-rate identity (2J_f+1) sum_mi 3j(J_f,J_i,1; -m_f, m_i, q)^2:")
for Ji, Jf in (
    (1, 0),
    (Sym(1) / 2, Sym(1) / 2),
    (Sym(3) / 2, Sym(1) / 2),
    (1, 1),
    (2, 1),
    (5, 5),
):
    good, bad = [], []
    mfs = [Jf - n for n in range(int(2 * Jf) + 1)]
    for mf in mfs:
        mis = [Ji - n for n in range(int(2 * Ji) + 1)]
        good.append(
            float(
                (2 * Jf + 1)
                * sum(wigner_3j(Jf, Ji, 1, -mf, mi, mf - mi) ** 2 for mi in mis)
            )
        )
        bad.append(
            float(
                (2 * Jf + 1)
                * sum(wigner_3j(Jf, Ji, 1, -mf, mi, mi - mf) ** 2 for mi in mis)
            )
        )
    print(
        f"  (J_i,J_f) = ({Ji},{Jf}): q = m_f - m_i -> {[round(x, 10) for x in good]}; "
        f"q = m_i - m_f -> {[round(x, 6) for x in bad]}"
    )

# straight vs reversed dark-state pairing under sigma+ light
print(
    "\nDark-state test for sigma+ light (E_{+1} = 0, E_{-1} = sqrt2), J_i=1 <-> J_f=0:"
)
E = {+1: 0.0, 0: 0.0, -1: np.sqrt(2)}
M = np.array([[E[mi] for mi in (-1, 0, +1)]])  # straight pairing: sum_m c_m E_m = 0
print(
    f"  straight pairing row [E_-1, E_0, E_+1]  = {M[0]}  -> dark space = span{{|1,0>, |1,+1>}} (c_-1 = 0)"
)
Mr = np.array([[E[+1], E[0], E[-1]]])
print(
    f"  printed reversed row [E_+1, E_0, E_-1]  = {Mr[0]}  -> would give span{{|1,-1>, |1,0>}} (wrong)"
)


# ----------------------------------------------------------------------------------
sep("2. E2 prefactor, end-to-end 40Ca+ D-state lifetimes (Kreuter Eqs. 7-8)")
# ----------------------------------------------------------------------------------
a0_A = a0_m * 1e10
pref_15 = (1 / 15) * (2 * pi * a0_A) ** 5 / t_au
pref_75 = (1 / 75) * (2 * pi * a0_A) ** 5 / t_au
print(
    f"(1/15)(2 pi a0[A])^5 / t_au = {pref_15:.6e}   (printed 1.11995e18, ratio {pref_15 / 1.11995e18:.9f})"
)
print(
    f"(1/75) variant             = {pref_75:.6e}   (ratio to (1/15) form {pref_75 / pref_15:.6f} = 1/5 exactly)"
)
print(f"  a Q from the (1/75) convention is sqrt(5) = {np.sqrt(5):.6f} larger")

A_M1_D32_S12 = 7.39e-11  # s^-1, Ali & Kim 1988
rows = []
for label, Q, dQ, lam_A, jv in (
    ("D_3/2", 7.939, 0.037, 7325.91, 1.5),
    ("D_5/2", 9.740, 0.047, 7293.48, 2.5),
):
    A = pref_15 / lam_A**5 * Q**2 / (2 * jv + 1)
    tau = 1 / A
    dtau = tau * 2 * dQ / Q
    rows.append((label, A, tau))
    print(
        f"{label}: Q = {Q}({dQ}) a.u., lambda_vac = {lam_A} A, 2j_v+1 = {int(2 * jv + 1)}: "
        f"A = {A:.6f} s^-1, tau = {1e3 * tau:.1f} ms, 2 dQ/Q -> +-{1e3 * dtau:.1f} ms"
    )
ratio = rows[1][1] / rows[0][1]
print(
    f"theory ratio tau(D_3/2)/tau(D_5/2) = A(D_5/2)/A(D_3/2) = {ratio:.6f}  (paper 1.0259(9))"
)
print(
    f"experimental ratio 1176/1168 = {1176 / 1168:.6f} +- {np.sqrt((11 / 1168) ** 2 + (1176 * 9 / 1168**2) ** 2):.4f}"
)
print(
    f"single-channel justification: A(E2, D_3/2) / A_M1 = {rows[0][1] / A_M1_D32_S12:.4e} "
    f"({np.log10(rows[0][1] / A_M1_D32_S12):.1f} orders)"
)
print(
    f"measured decay constants: Gamma(D_5/2) = {1 / 1.168:.5f} s^-1 ({1 / 1.168 / (2 * pi):.4f} Hz), "
    f"Gamma(D_3/2) = {1 / 1.176:.5f} s^-1 ({1 / 1.176 / (2 * pi):.4f} Hz)"
)


# ----------------------------------------------------------------------------------
sep("3. Blackbody-stimulated M1 mixing between the Ca+ D levels (Kreuter Eq. 1)")
# ----------------------------------------------------------------------------------
A12, nu_DD = 2.45e-6, 1.82e12
Gam_D52 = 1 / 1.168
for T in (300.0, 299.3):
    x = h * nu_DD / (k_B * T)
    nbar = 1 / (np.expm1(x))
    W = A12 * nbar
    print(
        f"T = {T:6.1f} K: h nu/kT = {x:.5f}, n-bar = {nbar:.5f}, W_12 = A n-bar = {W:.4e} s^-1, "
        f"W/Gamma(D5/2) = {W / Gam_D52:.3e}, shift on 1168 ms = {1168 * W / Gam_D52:.4f} ms"
    )
x300 = h * nu_DD / (k_B * 300.0)
nbar300 = 1 / np.expm1(x300)
print(
    f"negative test: DIVIDING by the Bose factor instead of multiplying is low by n-bar^2 = {nbar300**2:.2f}x"
)
print(
    f"upward rate carries g_u/g_l = 6/4 = {6 / 4:.2f}, i.e. {1.5 * A12 * nbar300:.4e} s^-1 at 300 K"
)

# collisional quenching and j-mixing from the Knoop coefficients
G = {"q_H": 37e-12, "q_N": 170e-12, "j_H": 3e-10, "j_N": 13e-10}  # cm^3 s^-1
p_mbar, T = 1e-11, 300.0
n_cm3 = (p_mbar * 100.0) / (k_B * T) * 1e-6  # 1 mbar = 100 Pa; m^-3 -> cm^-3
Rq = (G["q_H"] + G["q_N"]) * n_cm3
Rj = (G["j_H"] + G["j_N"]) * n_cm3
print(
    f"\nn = p/(k_B T) at {p_mbar:.0e} mbar per partner, {T:.0f} K: n = {n_cm3:.4e} cm^-3; "
    f"R^q = {Rq:.3e} s^-1, R^j = {Rj:.3e} s^-1, total = {Rq + Rj:.3e} s^-1"
)
print(
    f"  source prints < 3e-4 s^-1: recomputation is {(Rq + Rj) / 3e-4:.2f}x that, so 3e-4 is a rounded bound, "
    f"not derived; either way total/Gamma_nat = {(Rq + Rj) / Gam_D52:.2e} < 1e-3"
)
print(
    f"  j-mixing / quenching coefficient ratio = {(G['j_H'] + G['j_N']) / (G['q_H'] + G['q_N']):.2f}x"
)


# ----------------------------------------------------------------------------------
sep("4. Reshelving offset and the lifetime-fit bias (Kreuter Eqs. 2-3)")
# ----------------------------------------------------------------------------------
tau_true, R = 1.168, 3e-3
Gam = 1 / tau_true
Gp = Gam + R
print(
    f"Gamma' = Gamma + R = {Gp:.6f} s^-1 (FASTER than Gamma = {Gam:.6f}); offset B = R/Gamma' = {R / Gp:.5e}"
)
t = np.linspace(0.025, 5.0, 400)
pD = (1 - R / Gp) * np.exp(-Gp * t) + R / Gp
popt, _ = curve_fit(lambda tt, T: np.exp(-tt / T), t, pD, p0=[tau_true])
sig = np.sqrt(np.clip(pD * (1 - pD), 1e-12, None))
poptw, _ = curve_fit(
    lambda tt, T: np.exp(-tt / T), t, pD, p0=[tau_true], sigma=sig, absolute_sigma=False
)
print(
    f"binomially weighted fit on the same grid: T = {1e3 * poptw[0]:.1f} ms, T - tau = "
    f"{1e3 * (poptw[0] - tau_true):+.1f} ms (the brief quotes +10 ms; the magnitude depends on the "
    f"weighting and on the actual waiting times, the sign does not)"
)
print(
    f"plain-exponential fit exp(-t/T) to A e^{{-Gamma' t}} + B on a 25 ms-5 s grid: "
    f"T = {1e3 * popt[0]:.1f} ms, T - tau = {1e3 * (popt[0] - tau_true):+.1f} ms "
    f"(sign is the point; magnitude is grid- and statistics-dependent, the source quotes -3 ms as its budget entry)"
)


# ----------------------------------------------------------------------------------
sep("5. Wineland 2003: Raman detuning optimum and the per-pi-pulse scattering cost")
# ----------------------------------------------------------------------------------
gam_2pi, wF_2pi, w0_2pi = 19.4e6, 198e9, 1.25e9
gam_over_wF = gam_2pi / wF_2pi
# ratio R_SE/|Omega| proportional to |x(x-1)| [1/x^2 + 2/(x-1)^2], x = Delta/omega_F
f_ratio = lambda x: abs(x * (x - 1)) * (1 / x**2 + 2 / (x - 1) ** 2)
r3 = minimize_scalar(
    f_ratio, bounds=(0.05, 0.95), method="bounded", options={"xatol": 1e-12}
)
print(
    f"argmin_x [ |x(x-1)| (1/x^2 + 2/(x-1)^2) ] = {r3.x:.6f} omega_F  (sqrt2 - 1 = {np.sqrt(2) - 1:.6f}); "
    f"f_min = {r3.fun:.6f} (2 sqrt2 = {2 * np.sqrt(2):.6f})"
)
f_bracket = lambda x: 1 / x**2 + 2 / (x - 1) ** 2
r4 = minimize_scalar(
    f_bracket, bounds=(0.05, 0.95), method="bounded", options={"xatol": 1e-12}
)
print(
    f"NEGATIVE TEST, minimizing the bracket ALONE: x = {r4.x:.6f} omega_F  "
    f"(omega_F/(1+2^(1/3)) = {1 / (1 + 2 ** (1 / 3)):.6f}) - a different, wrong answer"
)
print(
    f"Delta/2pi at the ratio optimum = {r3.x * wF_2pi / 1e9:.2f} GHz (source '~82 GHz')"
)
P_clock = 2 * np.sqrt(2) * pi * gam_over_wF
print(
    f"clock qubit |0,0>: R_SE/|Omega| = 4 sqrt2 gamma/omega_F = {4 * np.sqrt(2):.6f} gamma/omega_F; "
    f"P_SE = 2 sqrt2 pi gamma/omega_F = {2 * np.sqrt(2) * pi:.6f} gamma/omega_F = {P_clock:.4e} for 9Be+"
)
P_zeeman = (8 * pi / np.sqrt(6)) * gam_over_wF
print(
    f"|2,2> <-> |1,1>:  P_SE = 8 pi/sqrt6 gamma/omega_F = {8 * pi / np.sqrt(6):.6f} gamma/omega_F "
    f"= {P_zeeman:.4e}  (source '~0.001') - do NOT conflate the two ~1e-3 figures"
)
print(
    f"|delta_(0<->0)/Omega_(0<->0)| = 4 sqrt2 omega_0/omega_F = {4 * np.sqrt(2) * w0_2pi / wF_2pi:.3e} (table 3.6e-2)"
)
# the exact invariant delta = -(omega_0/gamma) R_SE, checked at random Delta and couplings
rng = np.random.default_rng(5)
worst = 0.0
for _ in range(200):
    D = rng.uniform(-3, 3) * wF_2pi
    if abs(D) < 1e-3 * wF_2pi or abs(D - wF_2pi) < 1e-3 * wF_2pi:
        continue
    br = 1 / D**2 + 2 / (D - wF_2pi) ** 2
    g2 = rng.uniform(0.1, 10)
    RSE = gam_2pi * g2 / 3 * br
    d00 = -(g2 * w0_2pi / 3) * br
    worst = max(worst, abs(d00 + (w0_2pi / gam_2pi) * RSE) / abs(d00))
print(
    f"invariant delta_(0<->0) = -(omega_0/gamma) R_SE: worst relative residual over 200 samples = {worst:.2e}"
)
# Table 1 reproduction
print("\nTable 1 (P_SE = 2 sqrt2 pi gamma/omega_F, |delta/Omega| = 4 sqrt2 nu_0/nu_F):")
table = (
    ("9Be+", 19.4e6, 0.198e12, 1.25e9, 8.7e-4, 3.6e-2),
    ("25Mg+", 43e6, 2.75e12, 1.79e9, 1.4e-4, 3.6e-3),
    ("43Ca+", 22.4e6, 6.7e12, 3.26e9, 3.0e-5, 2.8e-3),
    ("67Zn+", 76e6, 26.2e12, 7.2e9, 2.6e-5, 1.6e-3),
    ("87Sr+", 21.7e6, 24e12, 5.00e9, 8.0e-6, 1.2e-3),
    ("113Cd+", 44.2e6, 74e12, 15.2e9, 5.3e-6, 1.2e-3),
    ("199Hg+", 54.7e6, 274e12, 40.5e9, 1.8e-6, 8.4e-4),
)
for name, g, nF, n0, pse_p, ds_p in table:
    print(
        f"  {name:7s} P_SE = {2 * np.sqrt(2) * pi * g / nF:.3e} (printed {pse_p:.1e}); "
        f"|delta/Omega| = {4 * np.sqrt(2) * n0 / nF:.3e} (printed {ds_p:.1e})"
    )
# signed fine-structure detunings: Delta referenced to P_1/2, P_3/2 pathway at Delta - omega_F
D_opt = r3.x * wF_2pi
print(
    f"\nsigned detunings at the optimum: Delta/2pi = {D_opt / 1e9:+.2f} GHz (from 2P_1/2), "
    f"(Delta - omega_F)/2pi = {(D_opt - wF_2pi) / 1e9:+.2f} GHz (from 2P_3/2) - opposite signs, "
    f"which is the P_1/2/P_3/2 destructive interference omega_F/[Delta(Delta-omega_F)]"
)
print(
    f"  Zeeman sanity check for the Uys 9Be+ qubit: 2 x 13.996 GHz/T x 4.5 T = "
    f"{2 * 13.996 * 4.5:.1f} GHz vs quoted Omega_z/2pi = 124.1 GHz"
)


# ----------------------------------------------------------------------------------
sep("6. Uys dissipator: three collapse operators and the Gamma_el/4 prefactor")
# ----------------------------------------------------------------------------------
from qutip import Qobj, basis, sigmaz, sigmap, sigmam, mesolve, liouvillian, qeye

Gel, Gdu, Gud = 1.7, 0.3, 0.5
GRam = Gdu + Gud
# basis order (u, d): index 0 = |u>, index 1 = |d>; sigmap() = |u><d|
c_ops = [
    0.5 * np.sqrt(Gel) * sigmaz(),
    np.sqrt(Gdu) * sigmap(),
    np.sqrt(Gud) * sigmam(),
]
rho = Qobj([[0.4, 0.2 + 0.1j], [0.2 - 0.1j, 0.6]])
L = liouvillian(0 * qeye(2), c_ops)
drho = Qobj(
    np.reshape(L.full() @ np.reshape(rho.full(), (4, 1), order="F"), (2, 2), order="F")
)
print(
    f"d rho_uu/dt = {drho[0, 0].real:+.6f}   (-Gamma_ud rho_uu + Gamma_du rho_dd = {-Gud * 0.4 + Gdu * 0.6:+.6f})"
)
print(
    f"d rho_ud/dt = {drho[0, 1]:+.6f}   (-(1/2)(Gamma_Ram + Gamma_el) rho_ud = {-(GRam + Gel) / 2 * (0.2 + 0.1j):+.6f})"
)
print(f"trace derivative = {drho.tr().real:+.3e}")

# Rayleigh channel alone: pure dephasing at Gamma_el/2, populations untouched
psi0 = Qobj([[0.5, 0.5], [0.5, 0.5]])
ts = np.linspace(0, 4 / Gel, 401)
res = mesolve(0 * qeye(2), psi0, ts, c_ops=[0.5 * np.sqrt(Gel) * sigmaz()])
coh = np.array([abs(r[0, 1]) for r in res.states])
pop = np.array([r[0, 0].real for r in res.states])
rate = -np.polyfit(ts, np.log(coh), 1)[0]
print(
    f"\nRayleigh only, c = (1/2) sqrt(Gamma_el) sigma_z: fitted coherence decay rate = {rate:.6f} "
    f"(Gamma_el/2 = {Gel / 2:.6f}); population drift max = {abs(pop - 0.5).max():.2e}"
)
res2 = mesolve(0 * qeye(2), psi0, ts, c_ops=[np.sqrt(Gel / 2) * sigmaz()])
rate2 = -np.polyfit(ts, np.log(np.array([abs(r[0, 1]) for r in res2.states])), 1)[0]
print(
    f"NEGATIVE TEST, c = sqrt(Gamma_el/2) sigma_z: decay rate = {rate2:.6f} = {rate2 / (Gel / 2):.1f}x too fast - "
    f"the Gamma_el/4 dissipator prefactor is correct and must not be 'fixed'"
)
# spin-echo readout: short-time slope
tau = np.linspace(0, 1e-6 / (GRam + Gel), 50)
ruu = 0.5 * (1 - np.exp(-(GRam + Gel) * tau / 2))
print(
    f"spin-echo readout rho_uu = (1/2)(1 - e^{{-(Gamma_Ram+Gamma_el) tau/2}}): initial slope = "
    f"{np.polyfit(tau, ruu, 1)[0]:.6f} = (Gamma_Ram+Gamma_el)/4 = {(GRam + Gel) / 4:.6f}; "
    f"twice it is the paper's total decoherence rate {(GRam + Gel) / 2:.6f}"
)
# rejected rate-difference diagnostic
Guu, Gdd = 1.0, 1.0
print(
    f"rejected diagnostic Gamma_el,diff = 2(Guu-Gdd)^2/(Guu+Gdd) at Guu = Gdd = 1: {2 * (Guu - Gdd) ** 2 / (Guu + Gdd):.3f} "
    f"(identically zero by construction, which is the ~5x underprediction the source reports)"
)


# ----------------------------------------------------------------------------------
sep("7. g_F, the MHz/G trap, and the Breit-Rabi curvature (171Yb+)")
# ----------------------------------------------------------------------------------
g_J = 2.00225664
g_F = g_J / 2
mu_I_over_muN, I_spin = 0.4919, 0.5  # 171Yb nuclear moment, mu_N
g_I = -(mu_I_over_muN / (I_spin)) * m_e_over_m_p
print(f"g_F(2S_1/2, F=1) = g_J/2 = {g_F:.6f} (dimensionless)")
print(
    f"g_F mu_B/h = {g_F * mu_B_over_h / 1e10:.4f} x 1e10 Hz/T = {g_F * mu_B_over_h * 1e-4 / 1e6:.4f} MHz/G"
)
B = 5.9
print(
    f"delta_B/2pi at B = {B} G (adjacent m_F) = {g_F * mu_B_over_h * 1e-4 * B / 1e6:.3f} MHz (printed 8.2 MHz); "
    f"m_F = -1 to +1 span = {2 * g_F * mu_B_over_h * 1e-4 * B / 1e6:.2f} MHz"
)
print(
    f"NEGATIVE TEST, coding g_F = 1.4 as a dimensionless g-factor: "
    f"{1.4 * mu_B_over_h * 1e-4 * B / 1e6:.2f} MHz, a {100 * (1.4 / g_F - 1):.0f}% error in the quantity that gates the CPT window"
)
nu0 = 12.642812118466e9
c2 = (g_J - g_I) ** 2 * mu_B_over_h**2 / (2 * nu0) * 1e-8  # Hz/G^2
c2_gI0 = g_J**2 * mu_B_over_h**2 / (2 * nu0) * 1e-8
print(
    f"\ntaylor_c2 = (g_J - g_I)^2 mu_B^2/(2 h^2 nu_0) = {c2:.2f} Hz/G^2 "
    f"({c2_gI0:.2f} with g_I = 0), reproducing the quoted +310.8 Hz/G^2 and confirming it is "
    f"(1/2) d^2nu/dB^2, not d^2nu/dB^2"
)
for printed in (12.642815e9, 12.642819e9, 12.642821e9):
    dnu = printed - nu0
    print(
        f"  printed {printed / 1e9:.6f} GHz sits {dnu / 1e3:.3f} kHz above the zero-field value, "
        f"i.e. B = {np.sqrt(dnu / c2):.3f} G - a shifted value, never a zero-field constant"
    )


# ----------------------------------------------------------------------------------
sep(
    "8. Recoil kernel: second moments and the coherent-sum failure mode (Joshi Table A1)"
)
# ----------------------------------------------------------------------------------
p_sigma = {+1: np.sqrt(1 / 5), 0: np.sqrt(3 / 5), -1: np.sqrt(1 / 5)}
p_pi = {+1: np.sqrt(1 / 10), 0: np.sqrt(8 / 10), -1: np.sqrt(1 / 10)}
for name, p in (("Delta m_j = +-1 (sigma)", p_sigma), ("Delta m_j = 0 (pi)", p_pi)):
    s1 = sum(v**2 for v in p.values())
    s2 = sum(p[q] ** 2 * (q**2) for q in p)  # (k_q/k)^2 = q^2 for k_q in {-k,0,+k}
    coh = sum(p.values()) ** 2
    print(
        f"{name:24s}: sum_q p^2 = {s1:.10f}; sum_q p^2 (k_q/k)^2 = {s2:.10f}; "
        f"(sum_q p)^2 = {coh:.7f}  <- coherent-sum inflation"
    )


def alpha_pattern(kind, axis="z"):
    if kind == "pi":
        N = lambda th: 3 / (8 * pi) * np.sin(th) ** 2
    elif kind == "sigma":
        N = lambda th: 3 / (16 * pi) * (1 + np.cos(th) ** 2)
    else:
        N = lambda th: 1 / (4 * pi)
    proj = (
        (lambda th, ph: np.cos(th) ** 2)
        if axis == "z"
        else (lambda th, ph: (np.sin(th) * np.cos(ph)) ** 2)
    )
    return dblquad(lambda th, ph: N(th) * proj(th, ph) * np.sin(th), 0, 2 * pi, 0, pi)[
        0
    ]


print(
    f"quadrature: <cos^2 theta> for (1+cos^2)/2 (sigma) = {alpha_pattern('sigma'):.6f} (2/5); "
    f"for sin^2 (pi) = {alpha_pattern('pi'):.6f} (1/5); isotropic = {alpha_pattern('iso'):.6f} (1/3)"
)
print(
    "  three different patterns under one definition, not rival values; alpha = 1/3 is the semiclassical"
)
print(
    "  isotropic heating term, 2/5 and 1/5 are the dissipator's recoil second moments"
)
print(f"C_m^2 per excited sublevel: 1/3 (pi) + 2/3 (sigma) = {1 / 3 + 2 / 3:.6f}")
print(
    f"H_al normalization sqrt(1/3) = sqrt(2/3) x 1/sqrt2: {np.sqrt(1 / 3):.16f} vs {np.sqrt(2 / 3) / np.sqrt(2):.16f}"
)


# ----------------------------------------------------------------------------------
sep("9. Polarization-gradient cooling limits (Joshi Eqs. 7-10)")
# ----------------------------------------------------------------------------------
n_fixed = lambda xi: xi + 1 / (4 * xi) - 0.5
n_avg = lambda xi: 0.75 * xi + 5 / (8 * xi) - 0.5
rf = minimize_scalar(
    n_fixed, bounds=(0.05, 5), method="bounded", options={"xatol": 1e-13}
)
ra = minimize_scalar(
    n_avg, bounds=(0.05, 5), method="bounded", options={"xatol": 1e-13}
)
print(
    f"fixed phase (phi = 0): min <n_0> = {rf.fun:.10f} at xi = {rf.x:.10f} (exactly 1/2 at xi = 1/2)"
)
print(
    f"phase averaged:        min <n>   = {ra.fun:.10f} at xi = {ra.x:.10f} "
    f"(sqrt(15/8) - 1/2 = {np.sqrt(15 / 8) - 0.5:.10f} at xi = sqrt(5/6) = {np.sqrt(5 / 6):.10f})"
)
# phase averages and the W, H integrals
ph = np.linspace(0, 2 * pi, 200001)
print(
    f"<cos^2 2phi> = {np.trapezoid(np.cos(2 * ph) ** 2, ph) / (2 * pi):.10f} (1/2); "
    f"<cos^4 2phi> = {np.trapezoid(np.cos(2 * ph) ** 4, ph) / (2 * pi):.10f} (3/8); "
    f"<sin^2 2phi> = {np.trapezoid(np.sin(2 * ph) ** 2, ph) / (2 * pi):.10f} (1/2)"
)
xi = np.sqrt(5 / 6)
W = (16 / 9) * np.cos(2 * ph) ** 2 * xi
H = (2 / 9) * (8 * xi**2 * np.cos(2 * ph) ** 4 + 2 + np.sin(2 * ph) ** 2)
print(
    f"integral(H)/integral(W) - 1/2 at xi = sqrt(5/6): "
    f"{np.trapezoid(H, ph) / np.trapezoid(W, ph) - 0.5:.13f} vs closed form {n_avg(xi):.13f}"
)
# recoil heating identity, alpha = 1/3 against 2/5
for a in (1 / 3, 2 / 5):
    Hsc = (a / 3) * (1 - np.sin(2 * ph) ** 2) + (1 / 3) * (1 + np.sin(2 * ph) ** 2)
    target = (2 / 9) * (2 + np.sin(2 * ph) ** 2)
    print(
        f"  alpha = {a:.5f}: H_carr + H_sb vs the non-xi^2 part of eq.(8): max ratio deviation "
        f"{abs(Hsc / target - 1).max():.2e}"
    )
# fitting-model consistency: eq. (13) n_0 = 3/2 sqrt(5/6)
print(
    f"eq.(13) consistency: n_0 = (3/2) sqrt(5/6) = {1.5 * np.sqrt(5 / 6):.6f} = sqrt(15/8) = min<n> + 1/2"
)
# moving-gradient window at the Joshi operating point
print(
    f"moving-gradient window W < delta < omega_z at the single-ion point: "
    f"W ~ 6.6e4 s^-1 < delta = 2 pi x 60 kHz = {2 * pi * 60e3:.2e} rad/s < omega_z = 2 pi x 1088 kHz = {2 * pi * 1088e3:.2e} rad/s"
)
# xi round trip
Delta_J, wz = 2 * pi * 210e6, 2 * pi * 1088e3
print(
    f"xi round trip: s = 3 omega_z xi/Delta at xi = 1.35 -> s = {3 * wz * 1.35 / Delta_J:.5f} (source's 0.021(2))"
)


# ----------------------------------------------------------------------------------
sep("10. Three-axis Lamb-Dicke triple from one crossed beam pair (171Yb+, 369.5 nm)")
# ----------------------------------------------------------------------------------
m_Yb = 171 * atomic_mass
lam = 369.5e-9
k = 2 * pi / lam
proj = {"x": 0.5, "y": 0.5, "z": 1 / np.sqrt(2)}
freqs = {"x": 0.790e6, "y": 0.766e6, "z": 0.525e6}
etas = {}
for ax in ("x", "y", "z"):
    w = 2 * pi * freqs[ax]
    x0 = np.sqrt(hbar / (2 * m_Yb * w))
    etas[ax] = k * proj[ax] * x0
    print(
        f"eta_{ax} = (2 pi/lambda)({proj[ax]:.6f}) sqrt(hbar/2 m omega) = {etas[ax]:.5f}  (printed {[0.052, 0.053, 0.090]['xyz'.index(ax)]})"
    )
print(
    f"projection norm (1/2)^2+(1/2)^2+(1/sqrt2)^2 = {sum(v**2 for v in proj.values()):.10f} (exactly 1)"
)
m_exact = 170.9363302 * atomic_mass
print(
    "with the actual isotope mass 170.9363 u instead of 171 u nominal: eta = "
    + ", ".join(
        f"{k * proj[ax] * np.sqrt(hbar / (2 * m_exact * 2 * pi * freqs[ax])):.5f}"
        for ax in ("x", "y", "z")
    )
    + "  (the consolidation's 0.05201, 0.05282, 0.09023; the +0.02% mass shift is the whole difference)"
)
w = 2 * pi * freqs["x"]
print(
    f"NEGATIVE TESTS on the zero-point convention: hbar/(4 m omega) -> eta_x = "
    f"{k * 0.5 * np.sqrt(hbar / (4 * m_Yb * w)):.4f}; hbar/(m omega) -> {k * 0.5 * np.sqrt(hbar / (m_Yb * w)):.4f}"
)
print(
    f"Lamb-Dicke breakdown at the Doppler-cooled start: eta_z sqrt(nbar+1) = "
    f"{etas['z'] * np.sqrt(21):.3f} at nbar = 20, {etas['z'] * np.sqrt(23):.3f} at nbar = 22"
)
Gam_e = 2 * pi * 19.6e6
Isat = pi * h * c * Gam_e / (3 * lam**3)
print(
    f"I_sat = pi h c Gamma_e/(3 lambda^3) = {Isat * 1e-1:.2f} mW/cm^2 (Ejtemaee adopts 51, ratio {51 / (Isat * 1e-1):.4f})"
)
