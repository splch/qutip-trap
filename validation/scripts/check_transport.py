"""Run 5 transport / splitting / junction checks behind the `[recomputed here]` numbers of
PLAN.md Section 4.6 (transport, splitting, merging and junctions; milestone M12).

Recomputes, in the plan's conventions (Section 13): the volt-potential Coulomb constant kappa,
the splitting critical point (d_CP, omega_CP in both printed forms), the two-ion Hessian at the
critical point by numerical diagonalization, the final-well frequency and the factor-2 error,
the critical-tilt algebra, the 4/25-versus-2/25 acceleration prefactors, the transport excitation
budget prefactor and its sudden limit, the sine- and erf-ramp suppression factors, the
single-sided S_E round trip, the Bose-Einstein conversion of a fitted-temperature table, the
four-ion crystal span, the junction micromotion peak/rms pair, the junction-centre Laplace
identity and the DAC update-noise ladder.

Run:
  uv run --python 3.13 --with qutip --with numpy --with scipy --with sympy --with mpmath \
      python check_transport.py
"""

import math

import numpy as np
import mpmath as mp
import sympy as sp
from scipy.constants import (
    hbar,
    h,
    e,
    epsilon_0,
    pi,
    atomic_mass,
    Boltzmann,
)
from scipy.optimize import brentq, fsolve
from scipy.integrate import quad

mp.mp.dps = 40

M_BE9 = 9.0121822 * atomic_mass  # 9Be+ (Reichle design study, Bowler, Blakestad)
M_CA40 = 39.9625909 * atomic_mass  # 40Ca+ (Kaufmann, Walther, Sterk)
M_YB171 = 170.9363 * atomic_mass  # 171Yb+ (Pino)
# the sources write "the mass of 40Ca+" and "171 u"; the round values differ from the isotope
# masses by 0.09% and 0.04%, which is where the last printed digit of omega_CP, omega_f and l moves
M_CA40_ROUND = 40.0 * atomic_mass
M_YB171_ROUND = 171.0 * atomic_mass

KAPPA_V = e / (4 * pi * epsilon_0)  # volt-potential Coulomb constant, V m
KAPPA_E = e * KAPPA_V  # energy-form Coulomb constant, N m^2


def head(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100)


# ---------------------------------------------------------------------------------------------
head("1. Coulomb constants and the volts-versus-joules convention")

print(
    f"kappa   = e/(4 pi eps0)   = {KAPPA_V:.10e} V m       (Kaufmann Sec. 2.1; brief 1.4399645478e-9)"
)
print(
    f"e kappa = e^2/(4 pi eps0) = {KAPPA_E:.7e} N m^2   (energy form; brief 2.3070776e-28)"
)
print(f"ratio e*kappa_V/kappa_E   = {e * KAPPA_V / KAPPA_E:.16f}  (must be 1)")
print(
    "quartic normalization map: a = alpha_energy/2 = e alpha_volt, b = beta_energy/4 = e beta_volt;"
)
print(
    f"  omega_loc^2 identity: -2 alpha_energy/m == -4 e alpha_volt/m  ->  ratio "
    f"{(2 * (2 * 1.0)) / (4 * 1.0):.6f} (2*alpha_e = 4*e*alpha_v when alpha_e = 2 e alpha_v)"
)

# ---------------------------------------------------------------------------------------------
head("2. Splitting: trap-A quartic coefficient, critical point d_CP and omega_CP")

# Kaufmann Table 1, trap A, per-volt Taylor coefficients
alpha_C, beta_C = -3.0e6, 2.7e13
alpha_S, beta_S = 1.7e6, -3.0e13
alpha_O, beta_O = 1.0e6, 0.2e13
gamma_S, gamma_O = 11.0e2, 3.2e2
U_lim = 10.0

beta_CP = (
    beta_O + (beta_C / alpha_C) * alpha_S - beta_S - (beta_C / alpha_C) * alpha_O
) * U_lim
print(
    f"beta_CP (Eq. 34, U_lim = 10 V)          = {beta_CP:.5e} V/m^4   (brief 2.570e14)"
)

d_CP = (2 * KAPPA_V / beta_CP) ** 0.2
print(
    f"d_CP = (2 kappa/beta)^(1/5)             = {d_CP * 1e6:.4f} um        (brief 25.70 um)"
)

w_CP_printed = beta_CP**0.3 * np.sqrt(3 * e / M_CA40) * (2 * KAPPA_V) ** 0.2
w_CP_dform = np.sqrt(3 * e * beta_CP / M_CA40) * d_CP
print(
    f"omega_CP (Eq. 15 as printed)/2pi        = {w_CP_printed / (2 * pi) / 1e3:.3f} kHz     (brief 176.4 kHz; Table 1 prints 0.18 MHz)"
)
print(
    f"omega_CP = sqrt(3 e beta/m) d_CP  /2pi  = {w_CP_dform / (2 * pi) / 1e3:.3f} kHz"
)
print(
    f"  the two printed forms agree to        {abs(w_CP_printed / w_CP_dform - 1):.3e} relative"
)
w_CP_round = beta_CP**0.3 * np.sqrt(3 * e / M_CA40_ROUND) * (2 * KAPPA_V) ** 0.2
print(
    f"  with the source's round m = 40 u      = {w_CP_round / (2 * pi) / 1e3:.3f} kHz     (this is the brief's 176.38 kHz)"
)

# consistency with Eq. (9), the external-curvature frequency at x = d_CP/2
w_eq9 = np.sqrt((e / M_CA40) * 12 * beta_CP * (d_CP / 2) ** 2)
print(
    f"omega from Eq. (9) at x = d_CP/2 /2pi   = {w_eq9 / (2 * pi) / 1e3:.3f} kHz  (identity with Eq. 15)"
)

# the same identity at two beta values far from trap A (brief check 23)
for b_test in (1e12, 1e18):
    dc = (2 * KAPPA_V / b_test) ** 0.2
    a = np.sqrt((e / M_CA40) * 12 * b_test * (dc / 2) ** 2)
    bb = b_test**0.3 * np.sqrt(3 * e / M_CA40) * (2 * KAPPA_V) ** 0.2
    print(
        f"  beta = {b_test:.0e} V/m^4: Eq.(9) {a:.11e} rad/s, Eq.(15) {bb:.11e} rad/s, d_CP = {dc * 1e6:.9f} um"
    )

# ---------------------------------------------------------------------------------------------
head("3. Splitting: two-ion Hessian at the critical point (numerical diagonalization)")


def U_two_ion(x, beta, alpha, gamma, kappa=KAPPA_V):
    """Potential ENERGY of two like ions on the axis, in joules; x = (x1, x2), x1 < x2."""
    x1, x2 = x
    ext = e * (beta * x1**4 + alpha * x1**2 + gamma * x1) + e * (
        beta * x2**4 + alpha * x2**2 + gamma * x2
    )
    return ext + e * kappa / abs(x2 - x1)


def grad_two_ion(x, beta, alpha, gamma, kappa=KAPPA_V):
    x1, x2 = x
    d = x2 - x1
    g1 = e * (4 * beta * x1**3 + 2 * alpha * x1 + gamma) + e * kappa / d**2
    g2 = e * (4 * beta * x2**3 + 2 * alpha * x2 + gamma) - e * kappa / d**2
    return np.array([g1, g2])


def hessian_num(f, x, hstep):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xp = np.array(x, dtype=float)
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = hstep
            ej[j] = hstep
            H[i, j] = (
                f(xp + ei + ej) - f(xp + ei - ej) - f(xp - ei + ej) + f(xp - ei - ej)
            ) / (4 * hstep**2)
    return H


x_eq = fsolve(
    lambda x: grad_two_ion(x, beta_CP, 0.0, 0.0),
    [-d_CP / 2, d_CP / 2],
    full_output=False,
    xtol=1e-14,
)
print(
    f"numerical equilibrium at alpha = 0, gamma = 0: x = ({x_eq[0] * 1e6:.6f}, {x_eq[1] * 1e6:.6f}) um"
)
print(
    f"  separation {(x_eq[1] - x_eq[0]) * 1e6:.6f} um against d_CP = {d_CP * 1e6:.6f} um "
    f"(relative {abs((x_eq[1] - x_eq[0]) / d_CP - 1):.3e})"
)

H = hessian_num(lambda x: U_two_ion(x, beta_CP, 0.0, 0.0), x_eq, d_CP * 1e-4)
evals = np.linalg.eigvalsh(H / M_CA40)  # equal masses: mass-weighted Hessian is H/m
w_num = np.sqrt(np.sort(evals))
print(
    f"Hessian (J/m^2):  [[{H[0, 0]:.6e}, {H[0, 1]:.6e}], [{H[1, 0]:.6e}, {H[1, 1]:.6e}]]"
)
print(
    f"  analytic: diagonal 4 e beta d^2 = {4 * e * beta_CP * d_CP**2:.6e}, "
    f"off-diagonal -e beta d^2 = {-e * beta_CP * d_CP**2:.6e}"
)
print(
    f"omega_COM/2pi = {w_num[0] / (2 * pi) / 1e3:.4f} kHz   (omega_CP = {w_CP_dform / (2 * pi) / 1e3:.4f} kHz)"
)
print(f"omega_str/2pi = {w_num[1] / (2 * pi) / 1e3:.4f} kHz")
print(
    f"omega_str/omega_COM = {w_num[1] / w_num[0]:.7f}   against sqrt(5/3) = {np.sqrt(5 / 3):.7f}"
)
print(f"  the harmonic-crystal ratio sqrt(3) = {np.sqrt(3):.7f} does NOT hold here")
print(
    f"  omega_COM/omega_CP - 1 = {w_num[0] / w_CP_dform - 1:.3e}  (Coulomb does not couple to x_0)"
)
print(f"  Bowler's simulated minima 880/700 kHz = {880 / 700:.5f} for a different trap")

# the (x_0, d) coordinates with effective masses 2m and m/2
d2_x0 = 6 * e * beta_CP * d_CP**2
d2_d = 2.5 * e * beta_CP * d_CP**2
print(
    f"(x_0, d) form: d2(e Phi_tot)/dx_0^2 = 6 e beta d^2 = {d2_x0:.6e}, m_x0 = 2m -> "
    f"omega = {np.sqrt(d2_x0 / (2 * M_CA40)) / (2 * pi) / 1e3:.4f} kHz"
)
print(
    f"               d2(e Phi_tot)/dd^2   = (5/2) e beta d^2 = {d2_d:.6e}, m_d = m/2 -> "
    f"omega = {np.sqrt(d2_d / (M_CA40 / 2)) / (2 * pi) / 1e3:.4f} kHz"
)

# ---------------------------------------------------------------------------------------------
head("4. Splitting: final well, the factor-2 error, and d_f")

alpha_f = -1.70e7  # V/m^2, trap A step-3 voltages
w_f = np.sqrt(-4 * e * alpha_f / M_CA40)
w_f_bad = np.sqrt(-2 * e * alpha_f / M_CA40)
print(
    f"omega_f = sqrt(-4 e alpha_f/m)/2pi = {w_f / (2 * pi) / 1e6:.4f} MHz   (brief 2.038 MHz)"
)
print(
    f"factor-2 variant sqrt(-2 e alpha_f/m)/2pi = {w_f_bad / (2 * pi) / 1e6:.4f} MHz  (brief 1.441 MHz)"
)
print(
    f"  ratio = {w_f / w_f_bad:.7f} = sqrt(2)   -> a sqrt(2) error in every n = dE/(hbar omega_f)"
)
print(f"hbar omega_f = {hbar * w_f / e * 1e9:.4f} neV     (brief 8.430 neV)")
w_f_round = np.sqrt(-4 * e * alpha_f / M_CA40_ROUND)
print(
    f"  with m = 40 u: omega_f/2pi = {w_f_round / (2 * pi) / 1e6:.4f} MHz, factor-2 variant "
    f"{w_f_round / np.sqrt(2) / (2 * pi) / 1e6:.4f} MHz, hbar omega_f = {hbar * w_f_round / e * 1e9:.3f} neV "
    f"(the brief's 2.038 / 1.441 / 8.430)"
)
d_f = np.sqrt(-2 * alpha_f / beta_CP)
print(f"d_f = sqrt(-2 alpha_f/beta) = {d_f * 1e6:.2f} um   (brief 363.7 um)")
d_i = 3.0e-6
print(
    f"printed-quintic terminus 2 d_i - d_f at d_i = 3 um: {(2 * d_i - d_f) * 1e6:.1f} um (negative: ions cross)"
)
# pre-critical relation, quartic neglected
alpha_i = 1e7
print(
    f"pre-CP: omega^2 = 2 e alpha/m at alpha = 1e7 V/m^2 -> {np.sqrt(2 * e * alpha_i / M_CA40) / (2 * pi) / 1e6:.4f} MHz; "
    f"d_i = (kappa/alpha)^(1/3) = {(KAPPA_V / alpha_i) ** (1 / 3) * 1e6:.6f} um"
)
# Coulomb-sign test: the printed pairing gives a root, the flipped one none (brief check 27)
f_minus = lambda d: 2 * alpha_i * (d / 2) - KAPPA_V / d**2
f_plus = lambda d: 2 * alpha_i * (d / 2) + KAPPA_V / d**2
root = brentq(f_minus, 1e-9, 1e-3, xtol=1e-18, rtol=1e-15)
grid = np.logspace(-9, -3, 61)
print(
    f"  Coulomb-sign test: '-' form root at d = {root * 1e6:.10f} um (identity {(KAPPA_V / alpha_i) ** (1 / 3) * 1e6:.10f}); "
    f"'+' form min over d in [1 nm, 1 mm] = {np.min(f_plus(grid)):.4e} V/m > 0, no root at all"
)

# ---------------------------------------------------------------------------------------------
head(
    "5. Splitting: perturbative d(alpha), x_0(gamma) and the 4/25 versus 2/25 prefactors"
)

lin_printed = -(1 / 5) * (16 / (beta_CP**4 * KAPPA_V)) ** 0.2
lin_ident = -2 / (5 * beta_CP * d_CP)
quad_printed = (2 / 25) * (4 / (beta_CP**7 * KAPPA_V**3)) ** 0.2
quad_ident = 4 / (25 * beta_CP**2 * d_CP**3)
print(
    f"dd/dalpha|_CP  printed  = {lin_printed:.10e}   identity -2/(5 beta d_CP) = {lin_ident:.10e}"
)
print(
    f"Eq.(16) alpha^2 coeff   = {quad_printed:.10e}   identity 4/(25 beta^2 d_CP^3) = {quad_ident:.10e}"
)
print(f"16^(1/5)/5 = {16**0.2 / 5:.5f}   (the prefactor Eq. 23 drops; brief 0.34822)")
d2d_rel = (4 / 25) * (4 / (beta_CP**7 * KAPPA_V**3)) ** 0.2
d2d_perion = (2 / 25) * (4 / (beta_CP**7 * KAPPA_V**3)) ** 0.2
print(
    f"RELATIVE coordinate d2d/dalpha^2 = (4/25)(...)^(1/5) = {d2d_rel:.6e}   (brief 2.855e-16)"
)
print(
    f"PER-ION       d2x/dalpha^2 = +-(2/25)(...)^(1/5)     = {d2d_perion:.6e}   (exactly half; CORRECT as printed)"
)
print(
    f"  ratio = {d2d_rel / d2d_perion:.7f} (exactly 2); substituting 4/25 per ion doubles the per-ion kick"
)
print(f"  4/25 = {4 / 25:.4f}, 2/25 = {2 / 25:.4f}")
# chi self-consistency: Eq. (26) with 4/25 must reproduce printed Eq. (27)
adot = 1.0e11
chi_26 = (d2d_rel * adot**2) / (d_CP * w_CP_dform**2)
chi_27 = (
    (4 / 25)
    * (M_CA40 / (3 * e))
    * 2 ** (-0.2)
    * beta_CP ** (-1.8)
    * KAPPA_V ** (-1.2)
    * adot**2
)
print(
    f"chi from Eq.(26) with 4/25 = {chi_26:.10e};  printed Eq.(27) = {chi_27:.10e};  ratio - 1 = {chi_26 / chi_27 - 1:.3e}"
)
adot_chi1 = np.sqrt(1.0 / (chi_27 / adot**2))
print(f"chi = 1 at alpha_dot_CP = {adot_chi1:.3e} V m^-2 s^-1   (brief 3.32e11)")
xi2 = 0.1
dE_chi1 = (pi**2 / 8) * xi2 * M_CA40 * (beta_CP**4 * KAPPA_V) ** (-0.4) * adot_chi1**2
print(
    f"delta E at chi = 1 (Eq. 25, xi^2 = 0.1) = {dE_chi1 / e:.3e} eV per ion  (brief 1.71e-4 eV)"
)
print(
    f"  -> n_bar = {dE_chi1 / (hbar * w_f):.3e} against hbar omega_f, {dE_chi1 / (hbar * w_CP_dform):.3e} against hbar omega_CP"
)
print(
    f"printed-chain coefficient pi^2/32 = {pi**2 / 32:.6f} versus printed Eq.(25) pi^2/8 = {pi**2 / 8:.6f}, "
    f"ratio {(pi**2 / 32) / (pi**2 / 8):.4f}"
)
x0_lead_printed = -1 / (3 * 2 ** (2 / 5) * beta_CP**0.6 * KAPPA_V**0.4)
print(
    f"x_0/gamma leading coeff printed = {x0_lead_printed:.9e} m per (V/m);  identity -1/(3 beta d_CP^2) = "
    f"{-1 / (3 * beta_CP * d_CP**2):.9e}"
)
a_edge = 0.1 * beta_CP * d_CP**2
print(
    f"quadratic/linear ratio of Eq.(16) at |alpha| = 0.1 beta d_CP^2: "
    f"{abs(quad_ident * a_edge**2 / (lin_ident * a_edge)):.6f}  (exactly 0.040 -> LINEAR-dominated)"
)
print(
    f"the two smallness conditions: kappa d_CP^-3 = {KAPPA_V / d_CP**3:.5e}, "
    f"beta d_CP^2/2 = {beta_CP * d_CP**2 / 2:.5e} V/m^2 (equal)"
)

# ---------------------------------------------------------------------------------------------
head("6. Splitting: critical-tilt bifurcation, well depth and the voltage tolerance")

u, x, gam, bet = sp.symbols("u x gamma beta", positive=True)
V = (
    bet * x**4
    - sp.Rational(3, 2) * bet ** sp.Rational(1, 3) * gam ** sp.Rational(2, 3) * x**2
    + gam * x
)
sub = {x: u * (gam / bet) ** sp.Rational(1, 3)}
s = gam ** sp.Rational(4, 3) * bet ** sp.Rational(-1, 3)
Vred = sp.simplify(sp.expand(V.subs(sub)) / s)
print(f"V/s in reduced coordinate u: {sp.simplify(Vred)}")
dV = sp.simplify(sp.diff(Vred, u))
print(f"dV/du = {sp.factor(dV)}   (roots: double at u = 1/2, simple at u = -1)")
Vsad = sp.nsimplify(Vred.subs(u, sp.Rational(1, 2)))
Vmin = sp.nsimplify(Vred.subs(u, -1))
print(
    f"V(saddle)/s = {sp.simplify(Vsad)}, V(minimum)/s = {sp.simplify(Vmin)}, "
    f"depth = {sp.simplify(Vsad - Vmin)} = {float(sp.simplify(Vsad - Vmin)):.6f}"
)
print(f"  27/16 = {27 / 16:.6f}; the printed subtraction order gives {-27 / 16:.6f}")
xx = sp.symbols("xx")
poly = sp.factor(16 * xx**4 - 24 * u**2 * xx**2 + 16 * u**3 * xx - 3 * u**4)
print(
    f"equal-value quartic factorizes as {poly}  -> x_tilde_c^(+) = -(3/2)(gamma/beta)^(1/3)"
)

gamma_tilde = 1.06 * (KAPPA_V**3 * beta_CP**2) ** 0.2
print(
    f"gamma_tilde = 1.06 (kappa^3 beta_CP^2)^(1/5) = {gamma_tilde:.4f} V/m   (brief 3.050; paper ~3 V/m)"
)
print(
    f"Delta U_O tolerance = gamma_tilde/gamma_O   = {gamma_tilde / gamma_O * 1e3:.2f} mV     (brief 9.53; paper ~9 mV)"
)
print(
    f"the paper's own saddle-balance heuristic gives C_gamma = (8/27)^(3/5) = {(8 / 27) ** 0.6:.5f}, "
    f"a factor {1.06 / (8 / 27) ** 0.6:.3f} smaller"
)
print(
    f"tilt at which one ion's CP curvature doubles: 0.787 (kappa^3 beta^2)^(1/5) = "
    f"{0.787 * (KAPPA_V**3 * beta_CP**2) ** 0.2:.4f} V/m = {0.787 / 1.06:.4f} gamma_tilde "
    f"(74%, NOT the source's ~67%)"
)
print(
    f"gamma_tilde scaling: proportional to beta_CP^(2/5), so the TOLERANCE grows with U_lim^(2/5) "
    f"(x{(5.0) ** 0.4:.3f} at U_lim = 50 V)"
)

# ---------------------------------------------------------------------------------------------
head("7. Splitting: the sign-corrected quintic distance ramp and the sin^2 ramp")

t, T = sp.symbols("t T", positive=True)
P_printed = -10 * t**3 / T**3 + 15 * t**4 / T**4 - 6 * t**5 / T**5
P_fixed = 10 * t**3 / T**3 - 15 * t**4 / T**4 + 6 * t**5 / T**5
for name, P in (("printed", P_printed), ("corrected", P_fixed)):
    vals = [
        float(P.subs({t: 0, T: 1})),
        float(P.subs({t: 1, T: 1})),
        float(sp.diff(P, t).subs({t: 0, T: 1})),
        float(sp.diff(P, t).subs({t: 1, T: 1})),
        float(sp.diff(P, t, 2).subs({t: 0, T: 1})),
        float(sp.diff(P, t, 2).subs({t: 1, T: 1})),
    ]
    print(
        f"{name:9s} quintic: P(0) = {vals[0]:+.1f}, P(T) = {vals[1]:+.1f}, "
        f"P'(0) = {vals[2]:+.1f}, P'(T) = {vals[3]:+.1f}, P''(0) = {vals[4]:+.1f}, P''(T) = {vals[5]:+.1f}"
    )
Psin = sp.sin(sp.pi * t / (2 * T)) ** 2
print(
    f"sin^2 ramp: P(T) = {float(Psin.subs({t: 1, T: 1})):+.1f}, "
    f"P''(0) T^2 = {float(sp.diff(Psin, t, 2).subs({t: 0, T: 1})):+.6f} = pi^2/2 = {pi**2 / 2:.6f} (non-vanishing)"
)

# ---------------------------------------------------------------------------------------------
head("8. Splitting: anomalous heating along the ramp, and its linear-in-T scaling")


def Gamma_h_trapA(f_MHz):
    return 6.3e3 * f_MHz ** (-1.81)  # s^-1 (the source writes 6.3 (f/MHz)^-1.81 per ms)


print(f"Gamma_h(1 MHz)    = {Gamma_h_trapA(1.0) / 1e3:.3f} /ms")
print(
    f"Gamma_h(0.18 MHz) = {Gamma_h_trapA(0.18) / 1e3:.1f} /ms   (brief 140/ms; Fig. 6 inset peak ~145-148/ms)"
)
print(
    f"implied noise exponent a = 1.81 - 1 = 0.81 (inside the reported 0.5 to 2.5 band)"
)
# a ramp fixed in t/T: Delta n_th = T * int_0^1 Gamma_h(omega(s)) ds, strictly linear in T
s_grid = np.linspace(0, 1, 2001)
f_ramp = 2.038 - (2.038 - 0.1764) * np.exp(
    -(((s_grid - 0.5) / 0.15) ** 2)
)  # illustrative dip to omega_CP
I = np.trapezoid(Gamma_h_trapA(f_ramp), s_grid)
for T_us in (6.0, 20.0, 70.0):
    print(
        f"  illustrative ramp, T = {T_us:4.0f} us: Delta n_th = T * I = {T_us * 1e-6 * I:.4f} quanta "
        f"(I = {I:.4e} s^-1)"
    )
print(
    "  Delta n_th is strictly PROPORTIONAL to T; the source's prose '1/T' contradicts its own Eq. (32) and Fig. 6"
)

# ---------------------------------------------------------------------------------------------
head("9. Splitting: trap-size scaling of the attainable quartic confinement")

w_seg, h_slit, d_spacer = 200e-6, 400e-6, 250e-6
d_eff = np.sqrt(w_seg**2 + h_slit**2 + d_spacer**2)
print(f"d_eff = (w^2 + h^2 + d^2)^(1/2) = {d_eff * 1e6:.2f} um")
print(
    f"implied SI prefactor beta_CP d_eff^4 = {beta_CP * d_eff**4:.2f} V   (brief 17.7 V)"
)
print(
    f"printed 2.2e24 V with d_eff in m  -> {2.2e24 * d_eff**-4:.3e} V/m^4 (23 orders too large)"
)
print(
    f"printed 2.2e24 V with d_eff in um -> {2.2e24 * (d_eff * 1e6) ** -4:.3e} V/m^4 (8x too small)"
)
print(
    "  -> use the d_eff^-4 EXPONENT only; the printed prefactor is unusable in every convention"
)

# ---------------------------------------------------------------------------------------------
head(
    "10. Transport: excitation-budget prefactor, the one-quantum threshold and the sudden limit"
)

b_dist, w0 = 400e-6, 2 * pi * 3.0e6
pref = M_BE9 * b_dist**2 * w0 / (8 * hbar)
print(
    f"m b^2 omega_0/(8 hbar), 9Be+, b = 400 um, 3 MHz = {pref:.5e}   (brief 5.3498e7)"
)
print(
    f"gamma <= 1 requires |Xi_tilde/omega_0|^2 <= {1 / pref:.5e}   (paper prints 2e-8, without the tilde)"
)
gamma_sudden = pref * 4.0
E_sudden = hbar * w0 * gamma_sudden
E_pot = 0.5 * M_BE9 * w0**2 * b_dist**2
print(
    f"sudden limit |Xi_tilde/omega_0| -> 2: gamma = {gamma_sudden:.4e}, hbar omega_0 gamma = {E_sudden:.6e} J"
)
print(
    f"  (1/2) m omega_0^2 b^2 = {E_pot:.6e} J   ratio {E_sudden / E_pot:.10f}   (brief 4.25374e-19 J)"
)
for bb, lab in ((1.2e-3, "1.2 mm"), (2.2e-3, "2.2 mm")):
    print(f"  prefactor at b = {lab}: {M_BE9 * bb**2 * w0 / (8 * hbar):.5e}")

# ---------------------------------------------------------------------------------------------
head(
    "11. Transport: sine ramp - closed form against quadrature, zeros, and the gamma = 1 crossing"
)


def xi_tilde_over_w0_numeric(theta_dd, x_val):
    """-e^{ix}(1/x) int_{-1}^{1} e^{-i x tau} theta''(tau) dtau  (Reichle Eqs. 43, 45, 51)."""
    re = quad(lambda tau: np.cos(-x_val * tau) * theta_dd(tau), -1, 1, limit=400)[0]
    im = quad(lambda tau: np.sin(-x_val * tau) * theta_dd(tau), -1, 1, limit=400)[0]
    return -np.exp(1j * x_val) * (re + 1j * im) / x_val


def sine_dd(tau):
    return -((pi / 2) ** 2) * np.sin(pi * tau / 2)


def sine_closed(x_val):
    return 2 * np.cos(x_val) / (1 - (2 * x_val / pi) ** 2)


print(
    "  x        |Xi~/w0| quadrature   |Xi~/w0| closed form    rel. diff    arg(Xi~/w0) - arg(-i e^{ix})"
)
for x_val in (0.3, 1.0, 3.0, 7.5, 20.0, 100.0):
    num = xi_tilde_over_w0_numeric(sine_dd, x_val)
    cf = sine_closed(x_val)
    phase_ref = -1j * np.exp(1j * x_val)
    dphi = np.angle(num / (phase_ref * abs(cf) * np.sign(cf)))
    print(
        f"{x_val:7.2f}   {abs(num):.12e}   {abs(cf):.12e}   {abs(abs(num) / abs(cf) - 1):.2e}   {dphi:+.2e} rad"
    )

print(
    "\nzeros of |Xi~|^2: cos x = 0 at x/2pi = (2n+1)/4; the n = 0 root x = pi/2 is cancelled by the"
)
print(
    "vanishing denominator 1 - (2x/pi)^2, so the surviving zeros are at x/2pi = (2n+3)/4:"
)
print("  " + ", ".join(f"{(2 * n + 3) / 4:.2f}" for n in range(6)))
for n in range(3):
    xz = 2 * pi * (2 * n + 3) / 4
    print(
        f"  x/2pi = {(2 * n + 3) / 4:.2f}: |Xi~/w0| closed = {abs(sine_closed(xz)):.3e}, "
        f"quadrature = {abs(xi_tilde_over_w0_numeric(sine_dd, xz)):.3e}"
    )
print(
    f"  at x = pi/2 the ratio is finite: |Xi~/w0| = {abs(xi_tilde_over_w0_numeric(sine_dd, pi / 2)):.6f} "
    f"(0/0 limit = pi/2 = {pi / 2:.6f})"
)


def sine_envelope_gamma(x_val, prefactor):
    return prefactor * 4 / (1 - (2 * x_val / pi) ** 2) ** 2


for bb, lab in ((400e-6, "400 um"), (1.2e-3, "1.2 mm")):
    pf = M_BE9 * bb**2 * w0 / (8 * hbar)
    xr = brentq(lambda xv: sine_envelope_gamma(xv, pf) - 1.0, 10.0, 1e5, xtol=1e-10)
    print(
        f"sine-ramp envelope crosses gamma = 1 at x/2pi = {xr / (2 * pi):.3f} for b = {lab} "
        f"(total 2 t_0/T = {2 * xr / (2 * pi):.1f} cycles)"
    )
print(
    "  brief: 30.238 (60 cycles) and 52.373 (~105 cycles); the paper writes 'x/2pi >~ 30' and 60 cycles"
)

# ---------------------------------------------------------------------------------------------
head("12. Transport: erf ramp - exact form, asymptote, and the paper's quanta figures")


def erf_closed_exact(x_val, y_val):
    """2 e^{-y^2/16} Re{Erf[2x/y + i y/4]} / Erf[2x/y]  (Reichle Eq. 49)."""
    r = mp.mpf(2) * x_val / y_val
    return float(
        2
        * mp.e ** (-(mp.mpf(y_val) ** 2) / 16)
        * mp.re(mp.erf(r + 1j * mp.mpf(y_val) / 4))
        / mp.erf(r)
    )


def erf_dd(tau, r):
    return -(4 * r**3 * tau / np.sqrt(pi)) * np.exp(-(r**2) * tau**2) / mp.erf(r)


print(
    "  y      x/2pi    |Xi~/w0| quadrature    Eq.(49) exact      4 e^{-y^2/8} asymptote   gamma(400 um)"
)
for y_val, x2pi in ((12.0, 4.0), (12.0, 8.0), (12.0, 20.0), (13.0, 4.0), (8.0, 6.0)):
    x_val = 2 * pi * x2pi
    r = 2 * x_val / y_val
    num = abs(xi_tilde_over_w0_numeric(lambda tau: float(erf_dd(tau, r)), x_val))
    ex = abs(erf_closed_exact(x_val, y_val))
    asym = 2 * np.exp(-(y_val**2) / 16)
    print(
        f"{y_val:5.1f}  {x2pi:6.1f}    {num:.9e}     {ex:.9e}    {asym:.9e}    {pref * num**2:.4f}"
    )

y12 = 12.0
print(f"\nasymptote squared 4 exp(-y^2/8) at y = 12: {4 * np.exp(-(y12**2) / 8):.6e}")
print(
    f"  gamma at b = 400 um: {pref * 4 * np.exp(-(y12**2) / 8):.4f}   (brief 3.259; the paper claims sub-quantum here)"
)
pref22 = M_BE9 * (2.2e-3) ** 2 * w0 / (8 * hbar)
xi13 = abs(erf_closed_exact(2 * pi * 4.0, 13.0))
print(
    f"exact Eq.(49) at y = 13, x/2pi = 4: |Xi~/w0| = {xi13:.5e}, gamma at b = 2.2 mm = {pref22 * xi13**2:.4f}"
)
print("  (brief 5.1681e-5 and 4.322; the paper claims 'less than a quantum')")
for pf, lab in ((pref, "400 um"), (pref22, "2.2 mm")):
    yr = brentq(lambda yv: pf * 4 * np.exp(-(yv**2) / 8) - 1.0, 4.0, 40.0, xtol=1e-12)
    print(f"  y giving gamma = 1 from the asymptote at b = {lab}: y = {yr:.3f}")
print(
    "  -> keep Reichle Eqs. (48)-(50); DO NOT use the paper's y = 12 / y = 13 quanta figures as targets"
)

# ---------------------------------------------------------------------------------------------
head("13. Transport: impulsive constant-velocity limit and the recapture nulls")

w_bowler = 2 * pi * 1.972e6
v = 370e-6 / 8e-6


def alpha_impulsive(t_T, w, vel, m):
    return np.sqrt(m * w / (2 * hbar)) * (1j * vel / w) * (1 - np.exp(-1j * w * t_T))


def alpha_quadrature(t_T, w, vel, m):
    re = quad(lambda tp: vel * np.cos(w * tp), 0, t_T, limit=400)[0]
    im = quad(lambda tp: vel * np.sin(w * tp), 0, t_T, limit=400)[0]
    return np.sqrt(m * w / (2 * hbar)) * (-np.exp(-1j * w * t_T)) * (re + 1j * im)


for t_T in (8.0e-6, 8.5e-6):
    a1 = alpha_impulsive(t_T, w_bowler, v, M_BE9)
    a2 = alpha_quadrature(t_T, w_bowler, v, M_BE9)
    print(
        f"t_T = {t_T * 1e6:.3f} us: closed form {a1.real:+.6f}{a1.imag:+.6f}j, quadrature "
        f"{a2.real:+.6f}{a2.imag:+.6f}j, |diff| = {abs(a1 - a2):.2e}"
    )
    print(f"            |alpha| = {abs(a1):.4f}, n_bar = {abs(a1) ** 2:.5e}")
t_null = 16 / 1.972e6
print(
    f"recapture null at omega t_T = 2 pi N, N = 16: t_T = {t_null * 1e6:.5f} us, "
    f"|alpha| = {abs(alpha_impulsive(t_null, w_bowler, v, M_BE9)):.3e}"
)
print(
    f"period count of the measured optimum: 1.972 MHz x 8 us = {1.972e6 * 8e-6:.3f} (paper: 'approximately N = 16')"
)
print(
    f"comb spacing in ordinary frequency 1/t_T = {1 / 8e-6 / 1e6:.4f} MHz (measured offsets 0.123-0.125 MHz)"
)
print(
    f"ground-state extent z_0 = sqrt(hbar/2 m omega) = "
    f"{np.sqrt(hbar / (2 * M_BE9 * w_bowler)) * 1e9:.2f} nm; sqrt(m omega/2 hbar) = "
    f"{np.sqrt(M_BE9 * w_bowler / (2 * hbar)):.4e} m^-1"
)

# ---------------------------------------------------------------------------------------------
head("14. Transport: Husimi-Kerner P_mn against the displaced-Fock matrix element")

try:
    from qutip import displace, basis, num, expect, variance

    N = 220
    ok = True
    for gval in (0.05, 0.5, 2.0, 7.0):
        alpha_c = np.sqrt(gval)
        D = displace(N, alpha_c)
        for n in (0, 1, 3, 5):
            psi = D * basis(N, n)
            probs = np.abs(psi.full().ravel()) ** 2
            mvals = np.arange(N)
            nu = np.maximum(mvals, n)
            mu = np.minimum(mvals, n)
            from scipy.special import eval_genlaguerre, gammaln

            closed = (
                np.exp(
                    gammaln(mu + 1) - gammaln(nu + 1) + (nu - mu) * np.log(gval) - gval
                )
                * eval_genlaguerre(mu, nu - mu, gval) ** 2
            )
            dmax = np.max(np.abs(probs - closed))
            if n == 0:
                pois = np.exp(-gval + mvals * np.log(gval) - gammaln(mvals + 1))
                dpois = np.max(np.abs(closed - pois))
            else:
                dpois = float("nan")
            mean = float(expect(num(N), psi))
            var = float(variance(num(N), psi))
            print(
                f"gamma = {gval:5.2f}, n = {n}: max|P_mn - |<m|D|n>|^2| = {dmax:.2e}, sum = {closed.sum():.10f}, "
                f"mean = {mean:.6f} (n + gamma = {n + gval:.6f}), var = {var:.6f} ((2n+1)gamma = {(2 * n + 1) * gval:.6f})"
                + (f", max|P_m0 - Poisson| = {dpois:.2e}" if n == 0 else "")
            )
            ok = ok and dmax < 1e-12
    print(f"all Husimi-Kerner closed forms agree with the displacement operator: {ok}")
except Exception as exc:  # pragma: no cover
    print(f"qutip unavailable or failed ({exc}); skipping the P_mn check")

# ---------------------------------------------------------------------------------------------
head(
    "15. Heating: the single-sided S_E round trip shared by splitting, transport and junctions"
)

w_z = 2 * pi * 3.6e6
ndot = 40.0
S_E = 4 * M_BE9 * hbar * w_z * ndot / e**2
print(
    f"S_E = 4 m hbar omega_z ndot/q^2, 9Be+ at 3.6 MHz, 40 quanta/s = {S_E:.4e} (V/m)^2/Hz"
)
print(
    f"  paper states 2.2e-13 at a 160 um ion-electrode distance; agreement {abs(S_E / 2.2e-13 - 1) * 100:.1f}%"
)
print(
    f"back-substitution Gamma_h = e^2 S_E/(4 m hbar omega) = {e**2 * S_E / (4 * M_BE9 * hbar * w_z):.4f} quanta/s"
)
for m_ion, f_MHz, rate, lab in (
    (M_CA40, 2.5, 295.0, "40Ca+ at 2.5 MHz, 295 quanta/s (Sterk)"),
    (M_YB171, 0.97, 300.0, "171Yb+ at 0.97 MHz, 300 quanta/s (Pino)"),
):
    ww = 2 * pi * f_MHz * 1e6
    print(f"  {lab}: S_E = {4 * m_ion * hbar * ww * rate / e**2:.3e} (V/m)^2/Hz")
print(
    f"anomalous heating over a 350 us E-C-E round trip at 40 quanta/s: {40.0 * 350e-6:.4f} quanta "
    f"(paper says ~0.007; the factor ~2 is consistent with omega_z ramping to 5.7 MHz)"
)
print(
    f"ambient heating over Sterk's 24 us round trip at 295 s^-1: {295.0 * 24e-6:.5f} quanta (paper: ~0.01)"
)
print(
    f"one quantum at 3.6 MHz = {h * 3.6e6 / e * 1e9:.3f} neV, so 0.1 quantum = {h * 3.6e6 / e * 1e8:.3f} neV "
    f"(the Table I caption's 1.6 neV is {1.6 / (h * 3.6e6 / e * 1e8) * 100 - 100:+.0f}% off)"
)

# ---------------------------------------------------------------------------------------------
head("16. Schedule: fitted temperatures to mean occupations (full Bose-Einstein)")


def nbar_BE(nu_MHz, nuT_MHz):
    return 1.0 / (np.exp(nu_MHz / nuT_MHz) - 1.0)


print(
    "  primitive          nu_T [MHz]   T [uK]    nu [MHz]   n_bar (Bose-Einstein)   nu_T/nu (Rayleigh-Jeans)   n_bar + 1/2"
)
rows = [
    ("swap, axial", 2.0, 0.97),
    ("swap, radial", 4.0, 2.7),
    ("swap, radial", 4.0, 2.8),
    ("split/combine, axial", 0.5, 0.97),
    ("split/combine, axial", 1.0, 0.97),
    ("split/combine, radial", 1.0, 2.7),
    ("split/combine, radial", 1.0, 2.8),
    ("shift, axial", 1.0, 0.97),
]
for lab, nuT, nu in rows:
    nb = nbar_BE(nu, nuT)
    print(
        f"  {lab:22s} {nuT:5.1f}   {h * nuT * 1e6 / Boltzmann * 1e6:7.1f}    {nu:5.2f}      "
        f"{nb:.4f}                {nuT / nu:.4f}                 {nb + 0.5:.4f}"
    )
print(
    "  the tabulated 'heat' columns are k_B T/h in MHz, not quanta; the Rayleigh-Jeans shortcut returns ~n_bar + 1/2"
)

# ---------------------------------------------------------------------------------------------
head("17. Crystal: axial length scale and the four-ion span")

w_x = 2 * pi * 0.97e6
ell = (e**2 / (4 * pi * epsilon_0 * M_YB171 * w_x**2)) ** (1 / 3)
ell_round = (e**2 / (4 * pi * epsilon_0 * M_YB171_ROUND * w_x**2)) ** (1 / 3)
print(
    f"l = (e^2/(4 pi eps0 m omega_x^2))^(1/3), 171Yb+ at 0.97 MHz = {ell * 1e6:.4f} um "
    f"(m = 170.9363 u); {ell_round * 1e6:.4f} um at the round 171 u (brief 2.7966 um)"
)


def eq_conditions(uu):
    out = np.zeros_like(uu)
    for i in range(len(uu)):
        s = 0.0
        for j in range(len(uu)):
            if i != j:
                s += np.sign(uu[i] - uu[j]) / (uu[i] - uu[j]) ** 2
        out[i] = uu[i] - s
    return out


u4 = fsolve(eq_conditions, np.array([-1.4, -0.45, 0.45, 1.4]), xtol=1e-14)
print(
    f"N = 4 dimensionless equilibria: {np.array2string(np.sort(u4), precision=5)}   (brief +-0.45438, +-1.43680)"
)
span = u4.max() - u4.min()
print(
    f"span = {span:.5f} l = {span * ell * 1e6:.4f} um (170.9363 u); {span * ell_round * 1e6:.4f} um at 171 u   "
    f"(brief 2.87360 l = 8.036 um; paper: '8 um')"
)
print(
    "  chain LENGTH is mass-independent at fixed dc curvature; only the mode spectrum differs for Ba-Yb-Yb-Ba"
)

# ---------------------------------------------------------------------------------------------
head("18. Junction: pseudopotential barrier, residual micromotion, peak versus rms")

Om_rf = 2 * pi * 83e6
V_rf = 200.0  # peak
phi_ps_zoneE = 2.9e-5  # volts (q phi_ps = 2.9e-5 eV)
E0 = np.sqrt(4 * M_BE9 * Om_rf**2 * phi_ps_zoneE / e)
z1_peak = e * E0 / (M_BE9 * Om_rf**2)
z1_peak_alt = 2 * np.sqrt(e * phi_ps_zoneE / (M_BE9 * Om_rf**2))
print(f"q phi_ps = 2.9e-5 eV  ->  E_0 = {E0:.1f} V/m   (brief 1.72e3 V/m)")
print(
    f"z_1^peak = q E_0/(m Omega_rf^2) = {z1_peak * 1e9:.2f} nm   (identity 2 sqrt(q phi_ps/(m Omega^2)) = "
    f"{z1_peak_alt * 1e9:.2f} nm)"
)
print(
    f"z_1^rms  = z_1^peak/sqrt(2)     = {z1_peak / np.sqrt(2) * 1e9:.2f} nm   (the paper quotes 47 nm, the rms-field value)"
)
print(
    "  a code path returning 68 nm from the stated 2.9e-5 eV is correct in the PEAK convention, not buggy"
)
grad_static = 8.7e-8 * e / 1e-6  # eV/um -> J/m
w_ax = 2 * pi * 3.6e6
print(
    f"static shift from the 8.7e-8 eV/um axial gradient in the 3.6 MHz well: "
    f"{grad_static / (M_BE9 * w_ax**2) * 1e9:.2f} nm (rules that out as the explanation)"
)

phi_ps_barrier = 0.3  # volts (0.3 eV for 9Be+)
grad_phi_rf = np.sqrt(phi_ps_barrier * 4 * M_BE9 * Om_rf**2 / (e * V_rf**2))
print(
    f"\n0.3 eV barrier for 9Be+ at 200 V peak, 2pi x 83 MHz requires |grad phi~_rf| = {grad_phi_rf:.1f} m^-1 "
    f"(brief 873 m^-1)"
)
print(
    f"  peak axial rf field at the apex = V_rf |grad phi~_rf| = {V_rf * grad_phi_rf:.3e} V/m  (brief 1.75e5)"
)
for m_ion, lab in ((M_CA40, "40Ca+"), (M_YB171, "171Yb+")):
    print(
        f"  barrier rescaled as 1/m at fixed charge and drive: {lab}: "
        f"{0.3 * M_BE9 / m_ion * 1e3:.1f} meV"
    )

# ---------------------------------------------------------------------------------------------
head("19. Junction centre: the traceless-rf-Hessian identity omega_y = 2 omega_x")

# near a 3D rf null E_rf = H r with H symmetric traceless; phi_ps curvature along axis i goes as h_i^2
h_x = 1.0
h_z = 1.0  # x/z channel equivalence
h_y = -(h_x + h_z)  # Laplace on the rf potential: trace zero
print(f"channel equivalence h_x = h_z = {h_x:.1f}  ->  Laplace forces h_y = {h_y:.1f}")
print(
    f"omega_rf,i proportional to |h_i|  ->  omega_y/omega_x = {abs(h_y) / abs(h_x):.4f}"
)
print(
    f"with omega_x/2pi = 5.7 MHz this predicts omega_y/2pi = {2 * 5.7:.1f} MHz against the reported 11.3 MHz "
    f"(ratio {11.3 / 5.7:.3f})"
)
print(
    "  the 5.7/5.7/11.3 MHz triple is self-consistent; the sum rule omega_x^2+omega_y^2+omega_z^2 = 2 omega_rf^2"
)
print(
    "  holds only where omega_rf,z = 0, i.e. in the straight channels and NOWHERE inside the junction"
)
w57 = 2 * pi * 5.7e6
w113 = 2 * pi * 11.3e6
print(
    f"  a naive 2 omega_rf^2 reading at C would need omega_rf/2pi = "
    f"{np.sqrt((2 * w57**2 + w113**2) / 2) / (2 * pi) / 1e6:.2f} MHz, which the pseudopotential does not supply"
)

# ---------------------------------------------------------------------------------------------
head("20. Junction: DAC update-noise resonance ladder J R_DAC = f_z")

f_z = 3.6e6
print("  J    R_DAC = f_z/J [kHz]   within the <= 500 kHz hardware cap?")
for J in range(7, 15):
    r = f_z / J
    print(
        f" {J:3d}      {r / 1e3:9.3f}            {'yes' if r <= 500e3 else 'NO (excluded)'}"
    )
R_DAC = 480e3
print(
    f"chosen R_DAC = 480 kHz gives f_z/R_DAC = {f_z / R_DAC:.3f}, exactly half-integer between J = 7 and J = 8"
)
print(
    "  observed resonances start at J = 8 because J = 7 would need 514.3 kHz, above the cap"
)
for L, dur, dwell, lab in (
    (1.76e-3, 350e-6, 20e-6, "E-C-E"),
    (3.52e-3, 910e-6, 20e-6, "E-C-F-C-E"),
    (2.84e-3, 950e-6, 20e-6, "E-C-V-C-E"),
):
    tmove = dur - dwell
    print(
        f"  {lab:10s}: {L * 1e3:.2f} mm in {tmove * 1e6:.0f} us -> {L / tmove:.2f} m/s, "
        f"{tmove * R_DAC:.1f} updates, {L / (tmove * R_DAC) * 1e6:.1f} um per update"
    )
print(
    "  the 5 um figure is the BEM/solution grid spacing; a 5 um/update rule would pin v at 2.4 m/s"
)
print(f"  and predict {1.76e-3 / 2.4 * 1e6:.0f} us for the 330 us E-C-E move")

# ---------------------------------------------------------------------------------------------
head("21. Adiabaticity parameters (all three require ANGULAR frequencies)")

print(
    f"Bowler's ramp caps: max|omega_dot/omega^2| = 0.025 (COM, harmonic ramp-down) and 0.015 "
    f"(around the sign change of the quadratic coefficient, no mode label)"
)
print(
    f"  substituting ordinary frequency changes these by 2 pi = {2 * pi:.4f} and destroys the numbers"
)
print(
    f"splitting chi = d_ddot_CP/(d_CP omega_CP^2), chi < 1 adiabatic; tau_CP = pi/omega_CP = "
    f"{pi / w_CP_dform * 1e6:.3f} us at trap A"
)
print(
    f"adiabatic suppression exp[c^2(1 - 1/chi)] with c UNDEFINED in the source; a reconstruction gives c ~ 0.93"
)
for Tt in (60e-6, 200e-6, 500e-6):
    # uniform sweep: alpha_dot ~ 1/T so chi ~ T^-2; delta E' ~ T^-2 exp(-c^2 const T^2), Gaussian in T
    chi = (60e-6 / Tt) ** 2
    print(
        f"  T = {Tt * 1e6:5.0f} us at chi(60 us) = 1: chi = {chi:.4f}, "
        f"suppression exp[0.93^2(1 - 1/chi)] = {np.exp(0.93**2 * (1 - 1 / chi)):.3e}"
    )
print(
    "  -> the adiabatic branch is T^-2 times a GAUSSIAN in T, not an exponential in T"
)

# ---------------------------------------------------------------------------------------------
head("22. Readout: Lamb-Dicke sideband coupling ladder and the pseudo-energy zero")

eta_s, g0 = 0.0613, 1.0
for m_ord in (1, 2):
    for n in (2, 3, 5):
        gm = (-1j * eta_s) ** m_ord / math.factorial(m_ord) * g0
        gnm = gm * np.sqrt(float(np.prod(range(n - m_ord + 1, n + 1))))
        if m_ord == 1:
            ref = -1j * eta_s * g0 * np.sqrt(n)
        else:
            ref = -0.5 * eta_s**2 * g0 * np.sqrt(n * (n - 1))
        print(
            f"m = {m_ord}, n = {n}: g_nm = {gnm:.12e}, printed special case {ref:.12e}, |diff| = {abs(gnm - ref):.2e}"
        )
eta_w = 0.23
M01 = eta_w * np.exp(-(eta_w**2) / 2)
P_plus = np.cos(M01 * 3 * pi / 2) ** 2
E_p = 2 * (P_plus - 1.0 - 1.0) + 3.5
print(
    f"\nWalther pseudo-energy at n = 0, eta = 0.23: M_{{0,+1}} = {M01:.5f}, P_{{up,+1}}(3pi) = {P_plus:.4f}, "
    f"E_p = {E_p:+.4f}"
)
eta_zero = brentq(
    lambda et: 2 * (np.cos(et * np.exp(-(et**2) / 2) * 3 * pi / 2) ** 2 - 2.0) + 3.5,
    0.1,
    0.3,
)
print(f"  E_p = 0 exactly at eta = {eta_zero:.5f} (2/9 = {2 / 9:.5f})")
print(
    f"  Debye-Waller 1 - eta^2 n at eta = 0.23, n = 20: {1 - eta_w**2 * 20:+.4f} (negative: the linearized "
    f"factor is invalid there)"
)
print(
    "  E_p is dimensionless, not quanta; single-valued only below ~50 quanta; the calibration curve is "
    "published only as a plot"
)

# ---------------------------------------------------------------------------------------------
head("23. Loss functional: the second weight sits INSIDE the square")

a1 = a2 = 2e-3  # Hz^-1  (2 kHz^-1)
I1, I2 = 1.5e3, 0.4e3  # illustrative Hz-valued sideband integrals
print(
    f"correct  L = a1 I1 + (a2 I2)^2 = {a1 * I1 + (a2 * I2) ** 2:.6f} (dimensionless)"
)
print(f"wrong    L = a1 I1 + a2 I2^2   = {a1 * I1 + a2 * I2**2:.6f} (carries Hz)")
print(
    f"  ratio of the second terms = {(a2 * I2) ** 2 / (a2 * I2**2):.3e} = a2 = {a2:.1e} Hz^-1"
)
nb = np.array([1.0, 10.0, 100.0, 1000.0])
print(
    "thermal envelope ordering (weak-probe limits): first-order sideband ~ <sqrt(n)> -> (sqrt(pi)/2)sqrt(n_bar), "
    "second-order ~ <sqrt(n(n-1))> -> n_bar"
)
for nn in nb:
    ns = np.arange(0, int(60 * nn) + 200)
    p = np.exp(ns * np.log(nn) - (ns + 1) * np.log(1 + nn))
    p = p / p.sum()
    print(
        f"  n_bar = {nn:7.1f}: <sqrt(n)>/sqrt(n_bar) = {np.sum(p * np.sqrt(ns)) / np.sqrt(nn):.5f} "
        f"(sqrt(pi)/2 = {np.sqrt(pi) / 2:.5f}), <sqrt(n(n-1))>/n_bar = "
        f"{np.sum(p * np.sqrt(ns * np.maximum(ns - 1, 0))) / nn:.5f}"
    )
print(
    "  so the alpha_1 term grows as sqrt(n_bar) and the squared alpha_2 term as n_bar^2: ratio ~ n_bar^(3/2)"
)
print(
    "second factorial moment: <n(n-1)> = n_bar^2 (Poissonian) versus 2 n_bar^2 (Bose-Einstein), a factor 2 "
    "at equal n_bar"
)

# ---------------------------------------------------------------------------------------------
head(
    "24. Distribution models: the independent-energy convolution versus the exact displaced thermal state"
)

from scipy.special import eval_laguerre, gammaln


def p_conv(n, nth, nal):
    ms = np.arange(0, n + 1)
    log_terms = (
        ms * np.log(nth)
        - (ms + 1) * np.log(nth + 1)
        - nal
        + (n - ms) * np.log(nal)
        - gammaln(n - ms + 1)
    )
    return float(np.sum(np.exp(log_terms)))


def p_dt(n, nth, nal):
    return (
        nth**n
        / (1 + nth) ** (n + 1)
        * np.exp(-nal / (1 + nth))
        * eval_laguerre(n, -nal / (nth * (1 + nth)))
    )


for nth, nal in ((0.1, 0.1), (0.1, 1.0)):
    pc = np.array([p_conv(n, nth, nal) for n in range(200)])
    pd = np.array([p_dt(n, nth, nal) for n in range(200)])
    ns = np.arange(200)
    vc = np.sum(pc * ns**2) - np.sum(pc * ns) ** 2
    vd = np.sum(pd * ns**2) - np.sum(pd * ns) ** 2
    print(
        f"n_th = {nth}, n_alpha = {nal}: p_0 conv {pc[0]:.6f} vs exact {pd[0]:.6f}, max|dp| = "
        f"{np.max(np.abs(pc - pd)):.3e}, sums {pc.sum():.8f}/{pd.sum():.8f}"
    )
    print(
        f"    variance conv {vc:.5f} vs exact {vd:.5f}; difference {vd - vc:.5f} against 2 n_th n_alpha = "
        f"{2 * nth * nal:.5f}"
    )
# the printed index range m = n..N keeps only the m = n term
nal = 1.0
print(
    f"printed index range m = n..N sums to exp(-n_alpha) = {np.exp(-nal):.5f} (a bare, unnormalised thermal "
    f"distribution); use m = 0..n"
)

print()
print("=" * 100)
print("done")
print("=" * 100)
