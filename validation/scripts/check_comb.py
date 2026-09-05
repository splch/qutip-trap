"""Checks behind the [recomputed here] numbers of PLAN.md Section 4.3.7 (frequency-comb
Raman drives, Run 5).

Everything here is 171Yb+ at 355 nm, the only system the Run 5 comb sources cover.
Conventions asserted (PLAN Section 13 and the new 4.3.7 rows):

  * internal angular (rad/s), public ordinary (Hz); nu_* is always Hz, omega_* always rad/s;
  * tau is the sech FIELD-envelope parameter in sech(pi t / tau) / sech(2 pi k nu_rep tau),
    never a FWHM;
  * a single tooth counted from the carrier carries sech(2 pi k nu_rep tau); a tooth PAIR
    indexed by separation l carries sech(pi l nu_rep tau).  The factor 2 is real physics;
  * Chain I of the intensity-to-Rabi calibration, Omega = g^2/(2 Delta), per the Section 13
    two-photon-Rabi row;
  * eta = Delta_k x_0 with x_0 = sqrt(hbar/(2 m omega)).

Sections printed:
  1  Dirichlet comb-tooth linewidth
  2  Tooth amplitudes, the intensity sum rule and the tooth-pair sums
  3  Rosen-Zener single-pulse ceilings and the one-sech identity
  4  Pulse-train-to-CW map and the (2 pi)^2 pulse-count defect
  5  Intensity-to-Rabi chain, signed fine-structure detunings, Chain I / Chain II E_pi
  6  Fourth-order (intensity-squared) ac Stark shift: single comb and the comb factor C
  7  Beat-note arithmetic: Hayes q, Islam lock, Inlek MS orders and PLL frequencies
  8  Lamb-Dicke parameters and the Hayes bare-k discriminator
  9  Comb noise channels: harmonic gain, per-tooth lock residuals, amplitude noise
 10  Spin-dependent kick: Jacobi-Anger, U_SDK algebra, the blue-sideband sqrt(n+1)
"""

import numpy as np
import mpmath as mp
from scipy.constants import h, hbar, c, epsilon_0, physical_constants
from scipy.special import jv
from scipy.optimize import brentq

u = physical_constants["atomic mass constant"][0]

# ---------------------------------------------------------------- species / laser constants
M_YB171 = 170.936323 * u  # 171Yb+ mass
NU_HF_0 = 12.642812118466e9  # zero-field 2S_1/2 clock splitting, Hz
NU_HF_LEE = 12.642821e9  # as printed by Lee 2016 (its comb-order arithmetic)
LAMBDA_355 = 355e-9
K_355 = 2 * np.pi / LAMBDA_355
GAMMA_P12 = 2 * np.pi * 19.6e6  # 2P_1/2 linewidth, ANGULAR
I_SAT_APB = 1500.0  # 0.15 W/cm^2, the paper-specific D1 convention, W/m^2


def sech(x):
    return 1.0 / np.cosh(np.clip(np.abs(x), 0.0, 700.0))


def head(n, title):
    print()
    print(f"== {n}. {title} " + "=" * max(0, 74 - len(title)))


# ============================================================ 1. Dirichlet tooth linewidth
head(1, "Dirichlet comb-tooth linewidth (Hayes Eq. 1: delta_nu ~ nu_R/N)")


def dirichlet_fwhm(N, power):
    """FWHM of |sin(N x/2)/sin(x/2)|^power in units of the tooth spacing 2 pi."""
    f = lambda x: abs(np.sin(N * x / 2) / np.sin(x / 2)) ** power / N**power
    half = brentq(lambda x: f(x) - 0.5, 1e-12, 2 * np.pi / N)
    return 2 * half / (2 * np.pi) * N  # in units of nu_R/N


for N in (64, 256, 1024):
    print(
        f"  N = {N:5d}:  amplitude FWHM = {dirichlet_fwhm(N, 1):.6f} nu_R/N,"
        f"   intensity FWHM = {dirichlet_fwhm(N, 2):.6f} nu_R/N"
    )
print(
    "  analytic half-widths: amplitude 1.895494/pi, intensity 1.391558/pi ->"
    f" {2 * 1.895494 / (2 * np.pi):.6f}, {2 * 1.391558 / (2 * np.pi):.6f} nu_R/N"
)
print("  the widely quoted 0.886 is the INTENSITY (power-spectrum) value")

# ============================================== 2. tooth amplitudes, sum rule, pair sums
head(
    2,
    "Tooth amplitudes g_k, intensity sum rule, tooth-pair sums (Lee Eq. 1; APB Eqs. 20-22)",
)

NU_REP_LEE, TAU_LEE = 120e6, 14e-12
a_lee = 2 * np.pi * NU_REP_LEE * TAU_LEE
kk = np.arange(-400_000, 400_001)
gk = np.sqrt(np.pi * NU_REP_LEE * TAU_LEE) * sech(a_lee * kk)  # g_k / g_0
print(
    f"  nu_rep = {NU_REP_LEE / 1e6:.0f} MHz, tau = {TAU_LEE * 1e12:.0f} ps, nu_rep*tau = {NU_REP_LEE * TAU_LEE:.3e}"
)
print(
    f"  sum rule   sum_k g_k^2 / g_0^2 = {np.sum(gk**2):.12f}   (asymptotic in nu_rep*tau, not algebraic)"
)
for x in (0.1, 1.0, 5.0):
    kx = np.arange(-4000, 4001)
    print(
        f"     at nu_rep*tau = {x:>4}: sum = {np.sum(np.pi * x * sech(2 * np.pi * x * kx) ** 2):.7f}"
    )

print(
    "  tooth-pair overlap  sum_k g_k g_{k+l} / g_0^2   vs   sech(pi l nu_rep tau)"
    "   vs the WRONG sech(2 pi l nu_rep tau):"
)
for l in (0, 10, 105):
    exact = np.sum(gk[: len(gk) - l] * gk[l:]) if l else np.sum(gk**2)
    right = sech(np.pi * l * NU_REP_LEE * TAU_LEE)
    wrong = sech(2 * np.pi * l * NU_REP_LEE * TAU_LEE)
    print(
        f"     l = {l:4d}:  exact {exact:.6f}   sech(pi l..) {right:.6f}   sech(2pi l..) {wrong:.6f}"
    )
print(
    "  -> at l = 105 the wrong argument gives 0.595 for 0.864, a 27% error;"
    " the closed form is itself ~5% high against the exact convolution"
)

# the two competing closed forms at the APB operating point
NU_REP_APB, TAU_APB = 80e6, 10e-12
omega_q = 2 * np.pi * NU_HF_0
n_apb = int(round(NU_HF_0 / NU_REP_APB))
a_apb = 2 * np.pi * NU_REP_APB * TAU_APB
gk_apb = np.sqrt(np.pi * NU_REP_APB * TAU_APB) * sech(a_apb * kk)
pair = np.sum(gk_apb[: len(gk_apb) - n_apb] * gk_apb[n_apb:])
x = omega_q * TAU_APB
print(
    f"  APB point (nu_rep = 80 MHz, tau = 10 ps, n = {n_apb}, omega_q tau = {x:.6f}):"
)
print(f"     tooth-pair sum          = {pair:.6f}")
print(f"     (omega_q tau)/sinh(..)  = {x / np.sinh(x):.6f}   [1 - x^2/6]")
print(
    f"     sech(omega_q tau/2)     = {sech(x / 2):.6f}   [1 - x^2/8]  <- APB Eq. 22, the form to implement"
)
print(
    f"     Hayes field-sech form (omega_q tau/2)/sinh(omega_q tau/2) at tau = 1 ps: "
    f"{(omega_q * 1e-12 / 2) / np.sinh(omega_q * 1e-12 / 2):.6f}   [1 - x^2/24]"
)
print(
    f"     gap between the first two: {100 * (sech(x / 2) / pair - 1):.2f}%  -- record, do not harmonize"
)

# ================================================= 3. Rosen-Zener ceilings, one-sech identity
head(3, "Rosen-Zener single-pulse ceiling and the ONE sech(omega_q tau/2) factor")

for T in (14.8e-12, 14e-12, 7.6e-12):
    print(
        f"  sech^2(pi nu_HF T) at T = {T * 1e12:4.1f} ps: {sech(np.pi * NU_HF_0 * T) ** 2:.6f}"
    )
print(
    f"  unit discriminator: nu_HF read as ANGULAR at 14.8 ps -> "
    f"{sech(np.pi * 2 * np.pi * NU_HF_0 * 14.8e-12) ** 2:.6f}  (against the measured 72%)"
)
tau_thr = 2 * np.arccosh(np.sqrt(2)) / omega_q
print(
    f"  two-pulse full-transfer threshold 2 arccosh(sqrt2)/omega_q = {tau_thr * 1e12:.3f} ps;"
    f" qubit period 1/nu_HF = {1 / NU_HF_0 * 1e12:.3f} ps"
)
print(
    f"  ONE factor, three faces, all equal at tau = 10 ps: sech(omega_q tau/2) = "
    f"{sech(omega_q * 10e-12 / 2):.6f}"
)
print(
    f"     applied twice -> {sech(omega_q * 10e-12 / 2) ** 2:.6f}, three times -> "
    f"{sech(omega_q * 10e-12 / 2) ** 3:.6f}  (spurious)"
)
print(
    f"  sech intensity FWHM = 2 arccosh(2)/pi = {2 * np.arccosh(2) / np.pi:.10f} tau;"
    f" a sech^2 intensity envelope gives {2 * np.arccosh(np.sqrt(2)) / np.pi:.4f} tau"
)

# ===================================================== 4. pulse-train-to-CW map, (2 pi)^2
head(4, "Pulse-train-to-CW map: N = f_rep t, Omega_0 = f_rep theta (APB Eq. 23 defect)")

Om_target = 2 * np.pi * 1e6  # the paper's worked Omega/2pi = 1 MHz
Om0 = Om_target / sech(omega_q * TAU_APB / 2)
theta = Om0 / NU_REP_APB
print(f"  worked example: Omega/2pi = 1 MHz at f_rep = 80 MHz, tau = 10 ps")
print(
    f"     Omega_0/2pi = {Om0 / 2 / np.pi / 1e6:.5f} MHz, theta = Omega_0/f_rep = {theta:.5f} rad"
)
print(
    f"     per-pulse Bloch angle varphi = theta sech(..) = {theta * sech(omega_q * TAU_APB / 2):.5f} rad"
    f"  -> pi pulse in {np.pi / (theta * sech(omega_q * TAU_APB / 2)):.1f} pulses"
)
print(
    f"  printed Omega_0 = 2 pi omega_rep theta would give theta = {Om0 / (2 * np.pi * 2 * np.pi * NU_REP_APB):.3e} rad,"
    f" too small by (2 pi)^2 = {(2 * np.pi) ** 2:.3f}"
)

# ============================================ 5. intensity-to-Rabi chain and Chain I / II
head(
    5,
    "Intensity-to-Rabi chain, signed fine-structure detunings, Chain I / Chain II E_pi",
)

D12, D32 = +33e12, -67e12  # SIGNED detunings at 355 nm, ordinary Hz
Deff = 1.0 / (1.0 / D12 - 1.0 / D32)
print(
    f"  1/Delta = 1/Delta_1/2 - 1/Delta_3/2 with {D12 / 1e12:+.0f} and {D32 / 1e12:+.0f} THz"
    f"  ->  Delta/2pi = {Deff / 1e12:.4f} THz"
)
print(
    f"     magnitudes-only misreading 1/(1/33 - 1/67) = {1 / (1 / 33e12 - 1 / 67e12) / 1e12:.3f} THz,"
    f" a factor {(1 / (1 / 33e12 - 1 / 67e12)) / Deff:.3f} in Delta and hence in Omega"
)

w = 10e-6
Delta_ang = 2 * np.pi * Deff
E_pi_II = np.pi**2 * I_SAT_APB * w**2 * Delta_ang / GAMMA_P12**2
print(
    f"  E_pi = pi^2 I_sat w^2 Delta / gamma^2 with I_sat = {I_SAT_APB:.0f} W/m^2, w = 10 um,"
    f" gamma and Delta ANGULAR:"
)
print(f"     Chain II (Omega = g^2/Delta,   as printed ~12 nJ): {E_pi_II * 1e9:.3f} nJ")
print(
    f"     Chain I  (Omega = g^2/2Delta,  the plan's Section 13 row): {2 * E_pi_II * 1e9:.3f} nJ"
)
E_pi_ord = np.pi**2 * I_SAT_APB * w**2 * Deff / (19.6e6) ** 2
print(
    f"     ordinary-frequency gamma and Delta instead: {E_pi_ord * 1e9:.1f} nJ  (the angular convention is load-bearing)"
)
print(
    f"  I_sat conventions: paper-specific D1 value {I_SAT_APB / 1e4 * 1e3:.0f} mW/cm^2 against the two-level"
    f" pi h c gamma/(3 lambda^3) at 369.5 nm =",
    end=" ",
)
I0_369 = np.pi * h * c * GAMMA_P12 / (3 * (369.5262e-9) ** 3)
print(f"{I0_369 / 1e4 * 1e3:.2f} mW/cm^2, a ratio {I_SAT_APB / I0_369:.4f}")
print(
    f"  Hayes' E_0 is FLUENCE-normalized: int|f|^2 dt = E_0^2 tau exactly while the peak is"
    f" f(0)/E_0 = sqrt(pi/2) = {np.sqrt(np.pi / 2):.8f}"
)
print(
    f"     feeding a measured PEAK field or intensity in as E_0 overstates Omega_0 and Ibar by"
    f" pi/2 = {np.pi / 2:.4f}"
)

# second-order differential shift and the D1/D2 near-cancellation at 355 nm
NU_D1, NU_D2 = c / 369.5262e-9, c / 328.937e-9
om_F = 2 * np.pi * (NU_D2 - NU_D1)
Del = 2 * np.pi * (NU_D1 - c / LAMBDA_355)  # comb centre BELOW P_1/2 in wavelength
Del = 2 * np.pi * (c / LAMBDA_355 - NU_D1)  # ... i.e. blue of P_1/2: Delta > 0
Pbar, w0 = 0.200, 3e-6
Ibar = 2 * Pbar / (np.pi * w0**2)
g0sq = GAMMA_P12**2 * Ibar / (2 * I0_369)
E2 = lambda D: (g0sq / 12) * (1 / D - 2 / (om_F - D))
d2 = E2(Del + omega_q) - E2(Del)
print(
    f"  fine structure omega_F/2pi = {(NU_D2 - NU_D1) / 1e12:.3f} THz; Delta/2pi at 355 nm ="
    f" {Del / 2 / np.pi / 1e12:.3f} THz"
)
print(
    f"  second order at 200 mW / 3 um waist (Ibar = 2 Pbar/(pi w_0^2) = {Ibar:.3e} W/m^2,"
    f" g_0/2pi = {np.sqrt(g0sq) / 2 / np.pi / 1e9:.1f} GHz):"
)
print(
    f"     E^(2)_00/2pi = {E2(Del) / 2 / np.pi / 1e3:+.1f} kHz  (a residual of 1/Delta ="
    f" {1 / Del:.4e} against 2/(omega_F - Delta) = {2 / (om_F - Del):.4e})"
)
print(
    f"     delta_omega^(2)/2pi = {d2 / 2 / np.pi / 1e3:+.2f} kHz   (Lee prints -7.3 kHz, an ORDINARY frequency)"
)
print(
    f"     with Pbar/(pi w_0^2) instead of the peak on-axis intensity:"
    f" {d2 / 2 / 2 / np.pi / 1e3:+.2f} kHz -- a factor-2 diagnostic"
)
print(
    f"  Delta_D1 = Delta_FS/3 is the exact zero of E^(2)_00 alone: lambda ="
    f" {c / (NU_D1 + (NU_D2 - NU_D1) / 3) * 1e9:.2f} nm"
)
r_at = lambda uu: (1 - uu) / uu + 2 * uu / (1 - uu)
u3, us = 1 / 3, np.sqrt(2) - 1
nuHF_over = NU_HF_0 / (2 * (NU_D2 - NU_D1))
print(
    f"     normalized differential shift (delta_0 - delta_1)/Omega_0,1: {nuHF_over * r_at(u3):.4e}"
    f" at Delta_FS/3, {nuHF_over * r_at(us):.4e} at (sqrt2 - 1) Delta_FS"
    f" (lambda = {c / (NU_D1 + us * (NU_D2 - NU_D1)) * 1e9:.1f} nm), i.e."
    f" {100 * (1 - r_at(us) / r_at(u3)):.1f}% lower -- '355 nm is the minimum' is approximate"
)

# ============================================= 6. fourth-order Stark shift and the comb factor
head(6, "Fourth-order (intensity-squared) ac Stark shift and the comb factor C_{n,a}")

mp.mp.dps = 30
integral = mp.quad(lambda v: mp.sech(v) ** 2 * mp.tanh(v) / v, [0, mp.inf])
closed = 7 * mp.zeta(3) / mp.pi**2
print(f"  int_0^inf sech^2(v) tanh(v)/v dv = {mp.nstr(integral, 12)}")
print(f"  7 zeta(3)/pi^2                   = {mp.nstr(closed, 12)}   (printed 0.853)")
print(
    f"  4 x that                         = {mp.nstr(4 * closed, 10)}   (printed 3.412, the two-comb smooth term)"
)

# single-comb Eq. 27 exact sum against the Eq. 28 closed form
Om0_1MHz = 2 * np.pi * 1e6
jj = np.arange(-4000, 4001)
jj = jj[jj != 0]
S27 = np.sum(sech((jj + n_apb) * 2 * np.pi * NU_REP_APB * TAU_APB / 2) ** 2 / jj)
d4_exact = -(Om0_1MHz**2 / (2 * 2 * np.pi * NU_REP_APB)) * S27
d4_closed = float(closed) * Om0_1MHz**2 * omega_q * TAU_APB / (2 * np.pi * NU_REP_APB)
print(
    f"  single comb at Omega_0/2pi = 1 MHz, tau = 10 ps, nu_rep = 80 MHz, n = {n_apb}:"
)
print(f"     Eq. 27 exact symmetrized sum: delta_4/2pi = {d4_exact / 2 / np.pi:.1f} Hz")
print(
    f"     Eq. 28 closed form:           delta_4/2pi = {d4_closed / 2 / np.pi:.1f} Hz   (printed +8.5 kHz)"
)
print(
    f"     effective coefficient of the exact sum = {float(closed) * d4_exact / d4_closed:.4f} against 0.85256"
    f"  (omega_q tau/2 = {omega_q * TAU_APB / 2:.3f} is not small)"
)
print(
    f"     with the PRINTED coefficient 0.853 instead of 7 zeta(3)/pi^2 the closed form gives "
    f"{d4_closed / float(closed) * 0.853 / 2 / np.pi:.1f} Hz, which is the brief's 8.470 kHz"
)


# ---- two-comb Stark-nulling AOM offset: Eq. 31 roots against the exact Eq. 30 line-2 sum
def bracket31(sigma, tau, nu_rep=NU_REP_APB):
    smooth = 4 * float(closed) * omega_q * tau
    return smooth + sech(omega_q * tau / 2) ** 2 * (
        1 / (2 * sigma) + 1 / (1 + 2 * sigma) + 1 / (2 * sigma - 1)
    )


def bracket30(sigma, tau, nu_rep=NU_REP_APB, jmax=20000):
    n = int(round(NU_HF_0 / nu_rep))
    a = np.pi * nu_rep * tau  # = omega_rep tau / 2
    j1 = np.arange(-jmax, jmax + 1)
    j1 = j1[j1 != 0]
    s1 = np.sum(sech((j1 + n) * a) ** 2 / j1)
    j2 = np.arange(-jmax, jmax + 1)
    s2 = np.sum(sech((j2 - n) * a) ** 2 / (j2 + 2 * sigma))
    return s1 - s2


print("  two-comb Stark-nulling offset sigma = omega_A mod omega_rep, over omega_rep:")
for tau_s in (5e-12, 10e-12):
    r31 = brentq(lambda s: bracket31(s, tau_s), 0.05, 0.499)
    r30 = brentq(lambda s: bracket30(s, tau_s), 0.05, 0.499)
    print(
        f"     tau = {tau_s * 1e12:4.0f} ps: Eq. 31 root sigma = {r31:.6f}"
        f"   exact Eq. 30 root sigma = {r30:.6f}   (paper quotes ~0.35 and ~0.40)"
    )
print(
    f"     tau -> 0 limits: Eq. 31 gives 1/sqrt(12) = {1 / np.sqrt(12):.6f}, the exact sum gives 1/4"
    "  -- root-find on the exact sum, the closed form can differ even in sign near a null"
)


def comb_factor_from(omega_a_ord, nu_rep, tau, kmax):
    """Return (j, delta_ord, C) with j = argmin_j |omega_a - 2 pi j nu_rep|."""
    j = int(round(omega_a_ord / nu_rep))
    delta_ord = omega_a_ord - j * nu_rep
    k = np.arange(-kmax, kmax + 1)
    num = sech((j + k) * np.pi * nu_rep * tau) ** 2
    den = 1.0 - k * nu_rep / delta_ord
    return j, delta_ord, float(np.sum(num / den))


nu_a_clock = NU_HF_LEE  # |00> -> |10>, ordinary Hz
j_c, d_c, _ = comb_factor_from(nu_a_clock, NU_REP_LEE, TAU_LEE, 10)
print(
    f"  nearest beat-note order for the clock transition at nu_rep = 120 MHz:"
    f" omega_HF/omega_rep = {nu_a_clock / NU_REP_LEE:.4f} -> j = {j_c}, residual delta_00,10/2pi = {d_c / 1e6:+.4f} MHz"
)
print(
    f"  k = 0 term alone = sech^2(j pi nu_rep tau) ="
    f" {sech(j_c * np.pi * NU_REP_LEE * TAU_LEE) ** 2:.6f}  (C is NOT normalized to 1)"
)
print(
    "  CONVERGENCE REGRESSION for C_00,10 -- assert the whole sweep, not successive agreement:"
)
for kmax in (0, 5, 10, 100, 500, 1000, 5000, 20000):
    _, _, C = comb_factor_from(nu_a_clock, NU_REP_LEE, TAU_LEE, kmax)
    print(f"     |k| <= {kmax:6d}:  C_00,10 = {C:.6f}")
_, _, C0010 = comb_factor_from(nu_a_clock, NU_REP_LEE, TAU_LEE, 20000)
_, _, C0010_10 = comb_factor_from(nu_a_clock, NU_REP_LEE, TAU_LEE, 10)
print(
    f"     the |k| ~ 5-10 plateau is a factor {C0010 / C0010_10:.3f} LOW against the converged value"
)

NU_ZEE = 7.000e6  # the source's rounded ~7 MHz at ~5 G
print(
    f"  (g_F mu_B/h x 5 G with g_F = g_J/2 = 1.001128 gives "
    f"{1.001128 * physical_constants['Bohr magneton'][0] / h * 1e-4 * 5 / 1e6:.4f} MHz; the rounded "
    f"7.000 MHz is used below, which is what the brief's C values assume)"
)
rows = [
    ("C_00,11 ", NU_HF_LEE + NU_ZEE),
    ("C_00,1-1", NU_HF_LEE - NU_ZEE),
    ("C_10,11 ", +NU_ZEE),
    ("C_10,1-1", -NU_ZEE),
]
for name, nu_a in rows:
    j, d, C = comb_factor_from(nu_a, NU_REP_LEE, TAU_LEE, 20000)
    print(f"  {name}: j = {j:4d}, delta/2pi = {d / 1e6:+8.3f} MHz, C = {C:.6f}")
print(
    "  -> j = 0 is the OPERATIVE case for the Zeeman terms and C_10,11 = C_10,1-1 exactly;"
    " a j != 0 guard deletes them and destroys the beta-hat cancellation.  Guard delta != 0."
)

# polarization branches and the sigma:beta ratio
_, d0011, C0011 = comb_factor_from(NU_HF_LEE + NU_ZEE, NU_REP_LEE, TAU_LEE, 20000)
_, d001m1, C001m1 = comb_factor_from(NU_HF_LEE - NU_ZEE, NU_REP_LEE, TAU_LEE, 20000)
ratio = (C0010 / d_c / 2) / ((C0011 / d0011 + C001m1 / d001m1) / 8)
print(
    f"  sigma_pm : beta-hat ratio of delta_omega^(4) = {ratio:.4f}  (Lee quotes 23/12 = "
    f"{23 / 12:.4f}); regress on this ratio, not on 23/12 MHz absolute"
)
print("  tau sensitivity of C_00,10 (the source's own tau-vs-bandwidth inconsistency):")
for t in (4e-12, 7e-12, 14e-12, 20e-12):
    _, _, C = comb_factor_from(nu_a_clock, NU_REP_LEE, t, 20000)
    print(f"     tau = {t * 1e12:4.0f} ps: C_00,10 = {C:.4f}")
print(
    f"  intensity-squared transfer: delta(delta_4)/delta_4 = 2 x delta(Omega)/Omega"
    f"  (delta_4 ~ Omega_0^2 ~ Ibar^2)"
)

# ================================================== 7. beat-note and lock arithmetic
head(
    7, "Beat-note arithmetic: Hayes q, Islam lock, Inlek MS orders and PLL frequencies"
)

for nu_R, label in (
    (80.78e6, "free-running"),
    (40.39e6, "picked 1-in-2"),
    (26.927e6, "picked 1-in-3"),
):
    print(
        f"  Hayes q = nu_0/nu_R at nu_R = {nu_R / 1e6:7.3f} MHz ({label:14s}): {12.6428e9 / nu_R:.4f}"
    )
print(
    "     -> 156.5 (half-integer, no evolution), 313.0 (integer, Rabi flopping), 469.5 (half-integer, none)"
)

nu_rep_islam, n_islam, nu_LO = 80.6e6, 157, 12.438e9
print(
    f"  Islam: {n_islam} x {nu_rep_islam / 1e6:.1f} MHz = {n_islam * nu_rep_islam / 1e9:.5f} GHz"
    f"  (paper ~12.655 GHz); nu_ab/nu_rep = {12.642819e9 / nu_rep_islam:.4f} -> n = 157"
)
print(
    f"     nu_M1 = n nu_rep - nu_LO = {(n_islam * nu_rep_islam - nu_LO) / 1e6:.3f} MHz (paper ~217 MHz)"
)
print(
    f"     nu_M2 = nu_ab - nu_LO    = {(12.642819e9 - nu_LO) / 1e6:.3f} MHz (paper ~205 MHz, the carrier)"
)
print(
    f"     |Delta nu_M| = {(n_islam * nu_rep_islam - 12.642819e9) / 1e6:.3f} MHz;"
    f" nu_sb = nu_LO + nu_M2 = nu_ab exactly (drift-free)"
)
print(f"     12.655 GHz / 157 = {12.655e9 / 157 / 1e6:.4f} MHz recovers nu_rep")

nu_r, nu_A, nu_0, nu_alpha, dlt = 80.57e6, 77.5e6, 12.64282e9, 2.5e6, 10e3
n_red, m_blue, p_car = 160, 154, 157
red_lhs, blue_lhs = nu_0 - nu_alpha + dlt, nu_0 + nu_alpha - dlt
nu_Br = n_red * nu_r - nu_A - red_lhs
nu_Bb = blue_lhs - m_blue * nu_r - nu_A
print(f"  Inlek Eq. 2 with n = {n_red} on the RED leg and m = {m_blue} on the BLUE:")
print(
    f"     nu_B,r = {nu_Br / 1e6:.3f} MHz (paper ~173.4), nu_B,b = {nu_Bb / 1e6:.3f} MHz (paper ~160.0)"
)
print(
    f"     swapping n and m misses each condition by"
    f" {abs((m_blue * nu_r - nu_A - nu_Br) - red_lhs) / 1e6:.2f} MHz -- a hard regression test"
)
nu_MO = 12.606e9
print(f"  Inlek Eq. A.1 at nu_MO = {nu_MO / 1e9:.3f} GHz:")
print(f"     nu_PLL1 = {n_red} nu_r - nu_MO = {(n_red * nu_r - nu_MO) / 1e6:.2f} MHz")
print(f"     nu_PLL2 = nu_MO - {m_blue} nu_r = {(nu_MO - m_blue * nu_r) / 1e6:.2f} MHz")
print(
    f"     carrier PLL3 = {p_car} nu_r - nu_MO = {(p_car * nu_r - nu_MO) / 1e6:.2f} MHz"
)
print(
    f"     single-PLL variant at 12.566 GHz: {(p_car * nu_r - 12.566e9) / 1e6:.2f} MHz"
)
awg_r = -nu_MO + nu_A + nu_0 - nu_alpha + dlt
awg_b = +nu_MO + nu_A - nu_0 - nu_alpha + dlt
print(
    f"     nu_AWG,r = {awg_r / 1e6:.2f} MHz, nu_AWG,b = {abs(awg_b) / 1e6:.2f} MHz;"
    f" the printed 116.8 / 43.2 MHz are each high by exactly 2 nu_alpha = {2 * nu_alpha / 1e6:.2f} MHz"
)
print(
    f"     loop closure: PLL1 - AWG_r = {(n_red * nu_r - nu_MO - awg_r) / 1e6:.2f} MHz = nu_B,r;"
    f" PLL2 - AWG_b = {(nu_MO - m_blue * nu_r - abs(awg_b)) / 1e6:.2f} MHz = nu_B,b"
)
print(
    f"  Inlek carrier (Eq. 1): nu_0/nu_r = {nu_0 / nu_r:.3f} -> p = {p_car},"
    f" nu_B,1 - nu_B,2 = {(nu_0 - p_car * nu_r) / 1e6:+.2f} MHz"
)
print(
    f"  Mizrahi time-domain delay T = n/(f_hf + f_a) at n = 5.5, f_a = 489 MHz:"
    f" {5.5 / (12.642815e9 + 0.489e9) * 1e12:.2f} ps (printed 419 ps)"
)
print(
    f"     hyperfine phase across it: omega_hf T = {2 * np.pi * 12.642815e9 * 5.5 / (12.642815e9 + 0.489e9):.2f} rad;"
    f" (omega_hf + omega_A) T = 2 pi x 5.5 = {2 * np.pi * 5.5:.3f} rad exactly"
)
lam_p = LAMBDA_355 / np.sqrt(2)
print(
    f"  effective two-photon wavelength at a 90 deg crossing: lambda/sqrt2 ="
    f" {lam_p * 1e9:.1f} nm"
)
for th_deg in (0.02, 0.05, 0.0133):
    th = np.deg2rad(th_deg)
    print(
        f"     wave-front tolerance Delta phi_M = 2 pi l sin(theta)/lambda' at l = 30 um,"
        f" theta = {th_deg:.4f} deg: {np.rad2deg(2 * np.pi * 30e-6 * np.sin(th) / lam_p):.2f} deg end-to-end"
    )
print(
    "     -> the source's own triple does not close (15.0 deg against a stated < 10 deg);"
    " theta < 0.0133 deg is what is needed, and the achieved 0.05 deg is 2.5x looser"
)

# Hayes gate self-consistency
t_g, nu_R_h, eta_h = 108e-6, 80.78e6, 0.1
dlt_h = 1 / t_g
Om_h = dlt_h / (2 * eta_h)  # ordinary Hz on both sides of delta = 2 eta Omega
print(
    f"  Hayes gate self-consistency: t_g = 108 us at nu_R = 80.78 MHz -> N = {t_g * nu_R_h:.0f} pulses;"
    f" delta/2pi = 1/t_g = {dlt_h / 1e3:.3f} kHz"
)
print(
    f"     inverting delta = 2 eta Omega: Omega/2pi = {Om_h / 1e3:.2f} kHz,"
    f" theta_p = Omega T = {2 * np.pi * Om_h / nu_R_h * 1e3:.3f} mrad, so theta_p << 1 and the"
    f" first-order truncation of Hayes Eq. 5 is justified"
)
print(
    f"     Hayes' eta Omega/delta = 1/2 is the J = (1/2) sum-sigma normalization;"
    f" halved on ingest to the plan's eta Omega/eps = 1/(4 sqrt K) (Section 13)"
)

# sech^2 envelope variation over one tooth index (why a flat 1:2:1 truncation is legitimate)
jj0 = j_c if "j_c" in dir() else 105
w_j = sech(jj0 * np.pi * NU_REP_LEE * TAU_LEE) ** 2
w_j1 = sech((jj0 + 1) * np.pi * NU_REP_LEE * TAU_LEE) ** 2
print(
    f"  sech^2 tooth-pair weight varies by {100 * (1 - w_j1 / w_j):.2f}% per unit tooth index at"
    f" nu_rep tau = {NU_REP_LEE * TAU_LEE:.2e}, which is what makes a flat few-tone truncation legitimate"
)

# ==================================================== 8. Lamb-Dicke parameters
head(8, "Lamb-Dicke parameters and the Hayes bare-k discriminator")


def eta_of(m, nu_t, dk):
    x0 = np.sqrt(hbar / (2 * m * 2 * np.pi * nu_t))
    return dk * x0, x0


for nu_t, dk, lbl in (
    (1.64e6, K_355, "Hayes 1.64 MHz, bare k"),
    (1.64e6, np.sqrt(2) * K_355, "Hayes 1.64 MHz, sqrt2 k (90 deg)"),
    (1.64e6, 2 * K_355, "Hayes 1.64 MHz, 2k (counterprop)"),
    (743e3, 2 * K_355, "Mizrahi 743 kHz, 2k"),
    (500e3, 2 * K_355, "Campbell 500 kHz, 2k"),
):
    e, x0 = eta_of(M_YB171, nu_t, dk)
    print(f"  {lbl:36s}: x_0 = {x0 * 1e9:.3f} nm, eta = {e:.4f}")
print(
    "  -> Hayes' quoted eta = 0.1 closes only with Delta_k = sqrt2 k, so its 'k' IS the two-photon Delta_k"
)
e500, _ = eta_of(M_YB171, 500e3, 2 * K_355)
print(
    f"  Lamb-Dicke guard at Campbell's nbar = 40: eta sqrt(nbar+1) = {e500 * np.sqrt(41):.3f} > 1"
    f"  -> a truncated expansion is illegal there"
)
print(
    f"  Hayes sideband-resolution threshold (omega_t T eta)^-1 at T = 12.4 ns, eta = 0.1:"
    f" omega_t T = {2 * np.pi * 1.64e6 * 12.4e-9:.4f}, threshold = {1 / (2 * np.pi * 1.64e6 * 12.4e-9 * 0.1):.1f}  ('N >> 80')"
)

# ==================================================== 9. comb noise channels
head(9, "Comb noise channels: harmonic gain, per-tooth lock residuals, amplitude noise")

print(
    f"  harmonic gain on the beat note: delta nu_sb = n delta nu_rep with n = 157:"
    f" 1 Hz/min -> {157 * 1:.0f} Hz/min = {157 / 60:.3f} Hz/s"
)
print(
    f"     over a 3 ms Ramsey window that is only {157 / 60 * 3e-3:.4f} Hz -- drift alone cannot explain"
    f" the ~3 ms unlocked coherence; the implied fast jitter is"
    f" {1 / (2 * np.pi * 3e-3):.1f} Hz rms on the beat note, {1 / (2 * np.pi * 3e-3) / 157:.3f} Hz rms on nu_rep [derived]"
)
print(
    f"     a 1e-7 fractional drift at 80 MHz: {1e-7 * 80e6:.1f} Hz x n = 158 ->"
    f" {1e-7 * 80e6 * 158 / 1e3:.2f} kHz at the qubit"
)
print("  per-tooth residuals under the Islam feed-forward lock (n = 157):")
for m_t in (156, 157, 158, 167):
    print(
        f"     tooth m = {m_t}: lower sideband {(m_t - 157):+4d} x delta nu_rep,"
        f" upper {(m_t + 157):+5d} x, bare {m_t:+5d} x"
    )
print(
    "     -> the lock nulls only the selected tooth's LOWER sideband; upper sidebands are worse than unlocked"
)

for a_db in (-90.0, -115.0, -120.0):
    al = 10 ** (a_db / 10)
    eps_hz = (np.pi**2 / 2) * al * (600e3) ** 2 * 1e-3
    eps_rad = (np.pi**2 / 2) * al * (2 * np.pi * 600e3) ** 2 * 1e-3
    print(
        f"  eps = (pi^2/2) alpha Omega_0^2 T at alpha = {a_db:7.1f} dB/Hz, Omega_0 = 600 kHz, T = 1 ms:"
        f"  Hz reading {eps_hz:.4e}   rad/s reading {eps_rad:.4e}"
    )
print(
    f"     ratio = (2 pi)^2 = {(2 * np.pi) ** 2:.3f}; structurally pi^2/2 = (2 pi)^2/8 = {np.pi**2 / 2:.10f}"
)
a_1pct = 10 * np.log10(1e-2 / ((np.pi**2 / 2) * (600e3) ** 2 * 1e-3))
print(
    f"     exact 1% boundary: alpha = {a_1pct:.2f} dB/Hz, so the quoted -115 dB/Hz carries"
    f" {a_1pct - (-115):.2f} dB of margin"
)
print(
    f"  rep-rate sensitivity of the fourth-order shift (a deliberately off-resonant Stark drive):"
)
base = comb_factor_from(nu_a_clock, NU_REP_LEE, TAU_LEE, 20000)
for frac in (1e-6, 1e-5, 1e-4):
    nr = NU_REP_LEE * (1 + frac)
    j2, d2, C2 = comb_factor_from(nu_a_clock, nr, TAU_LEE, 20000)
    s0 = base[2] / base[1]
    s1 = C2 / d2
    print(
        f"     dnu/nu = {frac:.0e}: d(delta_4)/delta_4 = {s1 / s0 - 1:+.4e}"
        f"  -> gain {(s1 / s0 - 1) / frac:.1f}x"
    )
print(
    f"     the detuning term alone gives j nu_rep/delta_00,10 ="
    f" {j_c * NU_REP_LEE / d_c:.1f}x; the rest is C's own nu_rep dependence"
)

# ================================================ 10. spin-dependent kick operators
head(
    10, "Spin-dependent kick: Jacobi-Anger, U_SDK algebra, the blue-sideband sqrt(n+1)"
)

try:
    from qutip import destroy, qeye, displace, sigmax, sigmap, sigmam, tensor

    Nf, eta_sdk, phi0 = 120, 0.2233, 0.83
    aop = destroy(Nf)
    # exact single exponential exp(i z sin(Delta_k x + phi) sigma_x)
    z = 1.17
    sinop = (
        np.exp(1j * phi0) * displace(Nf, 1j * eta_sdk)
        - np.exp(-1j * phi0) * displace(Nf, -1j * eta_sdk)
    ) / (2j)
    Hfull = tensor(sinop, sigmax())
    U_exact = (1j * z * Hfull).expm()
    U_sum = 0 * U_exact
    for n in range(-30, 31):
        sx_n = qeye(2) if n % 2 == 0 else sigmax()
        U_sum = U_sum + np.exp(1j * n * phi0) * jv(n, z) * tensor(
            displace(Nf, 1j * n * eta_sdk), sx_n
        )
    P = np.arange(60)  # compare on a low-n block
    da = np.abs(
        (U_exact - U_sum).full()[np.ix_(np.r_[P, Nf + P], np.r_[P, Nf + P])]
    ).max()
    U_sum_half = 0 * U_exact
    for n in range(-30, 31):
        sx_n = qeye(2) if n % 2 == 0 else sigmax()
        U_sum_half = U_sum_half + np.exp(1j * n * phi0) * jv(n, z / 2) * tensor(
            displace(Nf, 1j * n * eta_sdk), sx_n
        )
    db = np.abs(
        (U_exact - U_sum_half).full()[np.ix_(np.r_[P, Nf + P], np.r_[P, Nf + P])]
    ).max()
    print(
        f"  Jacobi-Anger, exponent coefficient z = {z}:  residual with J_n(z) = {da:.2e},"
        f" with J_n(z/2) = {db:.2e}"
    )
    print(
        "  -> the Bessel argument EQUALS the exponent coefficient; carry one symbol Theta_B defined as that"
    )
    print(
        f"  higher-order suppression at Theta_B = pi/8: J_2/J_1 = {jv(2, np.pi / 8) / jv(1, np.pi / 8):.4f}"
        f"  (n_max = 3-4);  at Theta_B = pi: {jv(2, np.pi) / jv(1, np.pi):.4f}  (truncation fails)"
    )

    # U_SDK algebra
    Dp, Dm = displace(Nf, 1j * eta_sdk), displace(Nf, -1j * eta_sdk)
    pp = 0.7
    for sgn, lbl in ((-1, "minus"), (+1, "plus ")):
        U = np.exp(1j * pp) * tensor(Dp, sigmam()) + sgn * np.exp(-1j * pp) * tensor(
            Dm, sigmap()
        )
        uni = np.abs((U.dag() * U - tensor(qeye(Nf), qeye(2))).full()).max()
        print(f"  U_SDK with the {lbl} relative sign: |U^dag U - 1| = {uni:.2e}")
    U = np.exp(1j * pp) * tensor(Dp, sigmam()) - np.exp(-1j * pp) * tensor(Dm, sigmap())
    print(
        f"     anti-Hermiticity |U^dag + U| = {np.abs((U.dag() + U).full()).max():.2e};"
        f" |U^2 + 1| = {np.abs((U * U + tensor(qeye(Nf), qeye(2))).full()).max():.2e}"
    )
    ex = (np.pi / 2 * U).expm()
    print(
        f"     |exp((pi/2) U) - U| = {np.abs((ex - U).full()).max():.2e}"
        f"  -> total train area pi, which is why sum_k Theta_B,k = pi"
    )

    # blue-sideband generator
    Nb = 8
    ab = destroy(Nb)
    Mgen = 1j * np.exp(1j * 0.0) * tensor(ab.dag(), sigmap()) - 1j * np.exp(
        -1j * 0.0
    ) * tensor(ab, sigmam())
    herm = np.abs((Mgen - Mgen.dag()).full()).max()
    diag = np.real(np.diag((Mgen * Mgen).full()))
    print(
        f"  blue-sideband generator M: |M - M^dag| = {herm:.2e};"
        f" diag(M^2) first entries = {np.round(diag[:8], 6).tolist()}"
    )
    print(
        "  -> M^2 = n_hat + 1 on one spin block and n_hat on the other, so the printed cos/sin closed form"
        " is exact only on the n = 0 manifold; the rate carries Theta eta sqrt(n+1)"
    )
except Exception as exc:  # pragma: no cover
    print(f"  [qutip section skipped: {exc}]")

print()
print("done.")
