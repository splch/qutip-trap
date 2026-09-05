"""Recomputations behind the 2026-09-04 four-lens critique fold (edit batch 19, PLAN.md Section 9.17).

Run:  uv run --python 3.13 --with qutip --with numpy --with scipy python check_critique_v3.py

Each block prints the number the plan quotes, with the alternative reading beside it where the critique found the plan
holding the wrong member of a pair. Sections: A sech envelope widths (4.3.7, 13); B saturation intensity with angular
Gamma (8.1, App. E); C dark-state destabilization optimum field (8.1); D Fock-tail populations of the GHZ fixtures
(9.6, 5.1.1); E zigzag threshold pair (9.1); F composite-pulse durations (4.3.5); G Monroe 1995 Lamb-Dicke anchor
geometry (4.2.2); H Morigi EIT fixture in the plan's sign convention (4.2.3, 4.2.8); I Floquet function sign under the
adopted Mathieu convention (4.1.1); J 171Yb+ quadratic Zeeman coefficient against g_J (4.5.1, 9.13); K filter-function
infrared cutoff (6.9, App. E); L order_slope round-off window (4.3.5, App. E); M QuTiP 5.3.1 probes (5.1.1, 5.3, 5.4,
11.2); N 355 nm second-order Stark weights (4.3.7); O the d_m = 8 row of the Section 5.1.1 displacement table.
"""

import warnings

import numpy as np
from scipy.constants import c, h, physical_constants, pi
from scipy.integrate import IntegrationWarning, quad, solve_ivp
from scipy.special import eval_genlaguerre, gammaln

warnings.filterwarnings("ignore", category=IntegrationWarning)
hbar = physical_constants["reduced Planck constant"][0]
mu_B = physical_constants["Bohr magneton"][0]
mu_N = physical_constants["nuclear magneton"][0]
u_kg = physical_constants["atomic mass constant"][0]


def head(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100)


# ---------------------------------------------------------------------------------------------
head(
    "A. sech envelope widths: the tooth formula sech(2 pi k nu_rep tau) belongs to a FIELD envelope sech(pi t/(2 tau))"
)
tau = 1.0
w_field_2tau = (4 / pi) * np.arccosh(2.0) * tau  # sech(pi t/2tau) = 1/2
w_int_2tau = (4 / pi) * np.arccosh(np.sqrt(2.0)) * tau  # sech^2(pi t/2tau) = 1/2
w_field_tau = (
    (2 / pi) * np.arccosh(2.0) * tau
)  # sech(pi t/tau) = 1/2  (the printed 0.8384)
w_int_tau = (
    (2 / pi) * np.arccosh(np.sqrt(2.0)) * tau
)  # sech^2(pi t/tau) = 1/2 (the printed 0.5611)
print(
    f"  field sech(pi t/(2 tau)):  FWHM_field = {w_field_2tau:.6f} tau,  FWHM_intensity(sech^2) = {w_int_2tau:.6f} tau"
)
print(
    f"  field sech(pi t/tau):      FWHM_field = {w_field_tau:.6f} tau,  FWHM_intensity(sech^2) = {w_int_tau:.6f} tau  (the printed pair)"
)


def ft_ratio(a, omegas):
    """|FT[sech(a t)](omega)| / |FT[sech(a t)](0)| by quadrature; exact value sech(pi omega/(2a))."""
    out = []
    f0 = quad(lambda t: 1 / np.cosh(a * t), -60 / a, 60 / a, limit=400)[0]
    for w in omegas:
        fw = quad(lambda t: np.cos(w * t) / np.cosh(a * t), -60 / a, 60 / a, limit=800)[
            0
        ]
        out.append(fw / f0)
    return np.array(out)


om = np.array([0.5, 1.0, 2.0, 3.0]) / tau
r2 = ft_ratio(pi / (2 * tau), om)
r1 = ft_ratio(pi / tau, om)
print("  FT check (ratio to omega = 0):")
for w, a, b in zip(om, r2, r1):
    print(
        f"    omega tau = {w * tau:3.1f}: FT[sech(pi t/2tau)] = {a:.6f} vs sech(omega tau) = {1 / np.cosh(w * tau):.6f};"
        f"  FT[sech(pi t/tau)] = {b:.6f} vs sech(omega tau/2) = {1 / np.cosh(w * tau / 2):.6f}"
    )
print(
    "  -> with teeth at omega = 2 pi k nu_rep, sech(2 pi k nu_rep tau) is the transform of sech(pi t/(2 tau)); the"
)
print(
    "     printed FWHM pair (0.8384, 0.5611) tau describes sech(pi t/tau), whose teeth would read sech(pi k nu_rep tau)."
)

# ---------------------------------------------------------------------------------------------
head("B. Saturation intensity I_sat = pi h c Gamma/(3 lambda^3) needs the ANGULAR rate")
lam = 369.5e-9
for name, g_hz in (
    ("partial S1/2 rate 19.62 MHz", 19.62e6),
    ("earlier 19.6 MHz reading", 19.6e6),
    ("total 19.72 MHz", 19.72e6),
):
    isat_ang = pi * h * c * (2 * pi * g_hz) / (3 * lam**3)
    isat_hz = pi * h * c * g_hz / (3 * lam**3)
    print(
        f"  {name:28s}: angular Gamma -> {isat_ang / 10:.2f} mW/cm^2;  gamma_hz fed in unconverted -> {isat_hz / 10:.2f} mW/cm^2 (ratio {isat_ang / isat_hz:.4f} = 2 pi)"
    )

# ---------------------------------------------------------------------------------------------
head(
    "C. Dark-state destabilization optimum (Berkeland-Boshier): delta_B = Omega/2 for a g_F ~ 1 level, Omega = gamma/3"
)
gamma_hz = 19.6e6
omega_rabi_hz = gamma_hz / 3  # Omega/2pi
delta_B_hz = omega_rabi_hz / 2
B_opt = delta_B_hz / (mu_B / h * 1e-4)  # mu_B/h in Hz per gauss
print(
    f"  delta_B/2pi = {delta_B_hz / 1e6:.3f} MHz -> B_opt = {B_opt:.2f} G (mu_B/h = {mu_B / h * 1e-4 / 1e6:.5f} MHz/G)"
)
print(
    f"  the 5 G operating point is {5 / B_opt:.2f}x the optimum (a factor about 2, not two decades); idealized g_J = 2, J = 1 level: {B_opt / 2:.2f} G"
)

# ---------------------------------------------------------------------------------------------
head(
    "D. Fock-tail populations of a coherent state |alpha| (top two levels of a d_m truncation), Poisson mean |alpha|^2"
)


def poisson_tail(mean, n_from):
    n = np.arange(0, 400)
    logp = -mean + n * np.log(mean) - gammaln(n + 1)
    p = np.exp(logp)
    return p[n >= n_from].sum()


for alpha in (0.5, 1.0, 1.5):
    row = []
    for d in (6, 8, 10, 12, 14):
        row.append(f"d_m={d:2d}: {poisson_tail(alpha**2, d - 2):.1e}")
    d_need = next(d for d in range(4, 60) if poisson_tail(alpha**2, d - 2) < 1e-6)
    print(
        f"  |alpha| = {alpha}: "
        + "  ".join(row)
        + f"  -> first d_m with top-two population < 1e-6: {d_need}"
    )
print(
    "  (a K = 1 MS gate displaces by up to 2 eta Omega/eps = 1/sqrt(K) = 1 per unit S_y eigenvalue pair; three aligned ions reach 1.5)"
)

# ---------------------------------------------------------------------------------------------
head(
    "E. Zigzag threshold: alpha_crit = 2/(mu_N - 1) against (omega_r/omega_z)_crit = 1/sqrt(alpha_crit)"
)
for N, mu_N_val in ((2, 3.0), (3, 29 / 5)):
    a_crit = 2 / (mu_N_val - 1)
    print(
        f"  N = {N}: mu_N = {mu_N_val:.4f}, alpha_crit = {a_crit:.6f}, (omega_r/omega_z)_crit = {1 / np.sqrt(a_crit):.6f}  (sqrt(12/5) = {np.sqrt(12 / 5):.6f})"
    )

# ---------------------------------------------------------------------------------------------
head("F. Composite-pulse durations at constant amplitude, tau = sum theta_l / Omega")
th = pi
k = np.arcsin(np.sin(th / 2) / 2)
corpse = (2 * pi + th / 2 - k) + (2 * pi - 2 * k) + (th / 2 - k)
print(f"  theta = pi, k = arcsin(sin(theta/2)/2) = {k:.6f} = pi/6")
print(
    f"  CORPSE  sum = {corpse / pi:.4f} pi = {np.degrees(corpse):.1f} deg   (4 pi + theta - 4k = {(4 * pi + th - 4 * k) / pi:.4f} pi)"
)
print(f"  SK1/BB1 sum = {(th + 4 * pi) / pi:.4f} pi                (4 pi + theta)")
print(
    f"  CinSK   sum = {(corpse + 4 * pi) / pi:.4f} pi = {np.degrees(corpse + 4 * pi):.1f} deg   (8 pi + theta - 4k = {(8 * pi + th - 4 * k) / pi:.4f} pi)"
)
print(
    f"  CinBB   sum = {(corpse + 4 * pi) / pi:.4f} pi                (same correctors: pi + 2 pi + pi)"
)
print(
    f"  printed (8 pi + 2 theta - 4k) = {(8 * pi + 2 * th - 4 * k) / pi:.4f} pi = {np.degrees(8 * pi + 2 * th - 4 * k):.1f} deg, {(8 * pi + 2 * th - 4 * k) / (corpse + 4 * pi) - 1:.1%} long"
)

# ---------------------------------------------------------------------------------------------
head(
    "G. Monroe 1995 Lamb-Dicke anchor (9Be+, 313 nm, modes 11.2/18.2/29.8 MHz; quoted eta = 0.21, 0.12, 0.09)"
)
m_be = 9.0121831 * u_kg
k313 = 2 * pi / 313e-9
for f_mhz, eta_q in ((11.2, 0.21), (18.2, 0.12), (29.8, 0.09)):
    x0 = np.sqrt(hbar / (2 * m_be * 2 * pi * f_mhz * 1e6))
    print(
        f"  {f_mhz:5.1f} MHz: x0 = {x0 * 1e9:.3f} nm; eta(|dk| = 2k) = {2 * k313 * x0:.4f}, eta(sqrt2 k, 90 deg crossing) = {np.sqrt(2) * k313 * x0:.4f},"
        f" eta(k) = {k313 * x0:.4f}; quoted {eta_q} -> implied |dk . e|/k = {eta_q / (k313 * x0):.3f}"
    )
print(
    "  -> the x anchor needs |dk| = sqrt2 k (a 90 degree crossing with dk along x), not 2k; the y and z anchors are"
)
print(
    "     reproduced to 8% and 3% by a projection k on each axis (a second 90 degree pair whose dk bisects y and z)."
)

# ---------------------------------------------------------------------------------------------
head(
    "H. Morigi EIT fixture (Fig. 3: nu = 2.0068, gamma = 20, Omega_1 = Omega_2 = 17 MHz, |Delta| = 70 MHz) in the plan's sign"
)
nu, gam, O1, O2, D = 2.0068, 20.0, 17.0, 17.0, 70.0
Or = np.sqrt(O1**2 + O2**2)
print(
    f"  Omega_r = sqrt(Omega_1^2 + Omega_2^2) = {Or:.4f} MHz; the EIT resonance condition Omega_r^2 = 4 nu (nu + Delta) gives {4 * nu * (nu + D):.2f} at Delta = +{D:.0f}"
)
print(
    f"  light shift of the bright state (sqrt(Delta^2 + Omega_r^2) - Delta)/2 = {(np.sqrt(D**2 + Or**2) - D) / 2:.4f} MHz = nu -> the fixture sits on resonance only for Delta > 0 in the plan's sign"
)


def n_eit(nu, gam, Or, D):
    """PLAN.md Section 4.2.3 closed form <n>_S = [gamma^2 nu^2 + 4(Omega_r^2/4 - nu(nu+Delta))^2]/[4 Delta nu (Omega_r^2 - 4 nu^2)]."""
    return (gam**2 * nu**2 + 4 * (Or**2 / 4 - nu * (nu + D)) ** 2) / (
        4 * D * nu * (Or**2 - 4 * nu**2)
    )


print(
    f"  Section 4.2.3 closed form: Delta = +70, Omega_r = 24.04 -> {n_eit(nu, gam, Or, D):.6f}  ((gamma/4Delta)^2 = {(gam / (4 * D)) ** 2:.6f});"
    f"  Delta = -70 (the caption's sign read in the plan's convention) -> {n_eit(nu, gam, Or, -D):.4f} (negative: trips the A_- - A_+ <= 0 guard);"
    f"  Delta = +70 with Omega_r = 17 (a single Rabi frequency in place of the quadrature sum) -> {n_eit(nu, gam, 17.0, D):.4f}, {n_eit(nu, gam, 17.0, D) / n_eit(nu, gam, Or, D):.0f}x the target"
)

# ---------------------------------------------------------------------------------------------
head(
    "I. Floquet function under the ADOPTED Mathieu sign d^2x/dxi^2 + [a - 2q cos 2xi] x = 0 (a = 0)"
)


def floquet(q, a=0.0, sign=-1):
    """Return beta, c_{+1}/c_0, c_{-1}/c_0 and u'(0)/u(0) for x'' + [a + sign*2q cos 2xi] x = 0."""

    def rhs(xi, y):
        return [y[1], -(a + sign * 2 * q * np.cos(2 * xi)) * y[0]]

    sol = solve_ivp(rhs, (0, pi), [1, 0], rtol=1e-12, atol=1e-14, dense_output=True)
    sol2 = solve_ivp(rhs, (0, pi), [0, 1], rtol=1e-12, atol=1e-14, dense_output=True)
    M = np.array([[sol.y[0, -1], sol2.y[0, -1]], [sol.y[1, -1], sol2.y[1, -1]]])
    ev, vec = np.linalg.eig(M)
    i = np.argmax(ev.imag)  # e^{+i beta pi}
    beta = np.angle(ev[i]) / pi
    v = vec[:, i]
    xs = np.linspace(0, pi, 2048, endpoint=False)
    u = v[0] * sol.sol(xs)[0] + v[1] * sol2.sol(xs)[0]
    up0 = v[1]  # u'(0) with u(0) = v[0]
    env = u * np.exp(
        -1j * beta * xs
    )  # periodic with period pi -> harmonics e^{2 i n xi}
    cn = np.fft.fft(env) / len(xs)
    c0, cp1, cm1 = cn[0], cn[1], cn[-1]
    return beta, cp1 / c0, cm1 / c0, up0 / v[0]


for q in (0.1, 0.2, 0.3):
    for sign, label in ((-1, "adopted a - 2q cos"), (+1, "RMP-quantum a + 2q cos")):
        beta, r1, rm1, ratio = floquet(q, sign=sign)
        pred = (
            sign * q / np.array([(2 + beta) ** 2, (2 - beta) ** 2])
        )  # c_{+-1}/c_0 = sign * q/((2 +- beta)^2 - a)
        print(
            f"  q = {q}, {label:24s}: beta = {beta:.6f}; c_+1/c_0 = {r1.real:+.6f} (pred {pred[0]:+.6f}), c_-1/c_0 = {rm1.real:+.6f} (pred {pred[1]:+.6f});"
            f" u'(0)/(i beta u(0)) = {(ratio / (1j * beta)).real:.6f} vs 1 {'+' if sign < 0 else '-'} q = {1 - sign * q:.4f}"
        )
print(
    "  -> under the adopted sign c_{+-1}/c_0 = -q/((2 +- beta)^2 - a): the lowest-order bracket is [1 - (q/2) cos(omega_rf t)] and"
)
print(
    "     u'(0) = i nu (1 + q) at u(0) = 1; the RMP's quantum section (W(t) with +2q cos) is the opposite rf phase origin,"
)
print(
    "     [1 + (q/2) cos] and i nu (1 - q). C_0 = 1 + 3q^2/16 is even in q and unaffected (check_c0_floquet.py)."
)

# ---------------------------------------------------------------------------------------------
head(
    "J. 171Yb+ clock-transition quadratic Zeeman coefficient (g_J - g_I)^2 mu_B^2/(2 h^2 nu_0) for the g_J values in the plan"
)
nu0 = 12.642812118466e9
mu_I_171 = 0.49367  # nuclear magnetons, Stone 2005 compilation (sign +)
I_171 = 0.5
g_I = -(mu_I_171 / I_171) * (
    mu_N / mu_B
)  # Steck sign: H = (mu_B/hbar)(g_J J + g_I I).B with g_I = -mu_I/(I mu_B)
for gJ, src in (
    (2.00225664, "43Ca+'s g_J, used by one row"),
    (2.00254, "one row, uncited"),
    (2.00292, "one row, uncited"),
    (2.002615, "Han et al. 2025, arXiv:2501.09973, ADOPTED"),
):
    coef = ((gJ - g_I) * mu_B * 1e-4 / h) ** 2 / (2 * nu0)
    coef0 = (gJ * mu_B * 1e-4 / h) ** 2 / (2 * nu0)
    print(
        f"  g_J = {gJ:.8f} ({src:40s}): {coef:.3f} Hz/G^2 with g_I = {g_I:+.3e}; {coef0:.3f} Hz/G^2 with g_I = 0"
    )
print(
    f"  Han et al. quote 31.0869(22) mHz/uT^2 = {31.0869 * 1e-3 / 1e-4:.3f} Hz/G^2; the g_J uncertainty 7e-5 moves the coefficient by {100 * 2 * 7e-5 / 2.0026:.4f}% = {310.87 * 2 * 7e-5 / 2.0026:.3f} Hz/G^2"
)
print(
    f"  spread over the three values the second revision carried: {100 * (((2.00292 - g_I) ** 2 - (2.00225664 - g_I) ** 2) / (2.00225664 - g_I) ** 2):.3f}% of the coefficient"
)

# ---------------------------------------------------------------------------------------------
head(
    "K. Filter-function infrared cutoff on a 1/omega^4 spectrum: omega_min = 1/T against 2 pi/T (T = 1, omega_max = 50)"
)
T = 1.0


def chi_fid(wmin, wmax=50.0):
    """(2/pi) int_{wmin}^{wmax} d omega omega^-4 F/omega^2 with F = 4 sin^2(omega T/2), free induction, unit spectrum."""
    return (
        2
        / pi
        * quad(lambda w: 4 * np.sin(w * T / 2) ** 2 / w**6, wmin, wmax, limit=400)[0]
    )


c1, c2 = chi_fid(1 / T), chi_fid(2 * pi / T)
print(
    f"  chi(omega_min = 1/T) = {c1:.6e}, chi(omega_min = 2 pi/T) = {c2:.6e}, ratio {c1 / c2:.1f}; the pure low-frequency estimate (2 pi)^3 = {(2 * pi) ** 3:.0f} holds only for omega_min T << 1"
)
print(
    f"  local sensitivity d ln chi / d ln omega_min at 2 pi/T: {(np.log(chi_fid(2 * pi / T * 1.01)) - np.log(chi_fid(2 * pi / T / 1.01))) / (2 * np.log(1.01)):.2f}; at 1/T: {(np.log(chi_fid(1.01 / T)) - np.log(chi_fid(1 / T / 1.01))) / (2 * np.log(1.01)):.2f}"
)
print(
    "  -> on an IR-divergent spectrum the reported chi is a function of the declared cutoff and must carry it."
)

# ---------------------------------------------------------------------------------------------
head(
    "L. order_slope window: 1 - F ~ c eps^{2(n+1)} against the ~4e-16 round-off floor of a 2x2 product"
)
for name, n, cc in (("primitive", 0, 2.47), ("SK1", 1, 22.83), ("BB1", 2, 9.39)):
    for eps in (1e-5, 1e-2, 1e-1):
        val = cc * eps ** (2 * (n + 1))
        print(
            f"  {name:10s} order {n}: eps = {eps:.0e} -> 1 - F ~ {val:.1e} {'(below round-off)' if val < 4e-16 else ''}"
        )

# ---------------------------------------------------------------------------------------------
head(
    "N. 355 nm second-order Stark weights: signed detunings against a double-counted line-strength sign"
)
D12, D32 = 33.0, -67.0  # THz, signed, Section 4.3.7
signed = 1 / D12 + 2 / D32
double = 1 / D12 - 2 / D32
print(
    f"  (1/Delta_1/2 + 2/Delta_3/2) = {signed:+.6f} THz^-1 (near-total D1/D2 cancellation); pairing -2 with the signed Delta_3/2 gives {double:+.6f} THz^-1: ratio {double / signed:.0f}, both positive, no sign flip"
)

# ---------------------------------------------------------------------------------------------
head(
    "O. Section 5.1.1 displacement table, the d_m = 8 row: expm elements against the analytic Laguerre elements"
)


def analytic_D(d, eta):
    """<n'|D(i eta)|n> = e^{-eta^2/2} sqrt(n_<!/n_>!) (i eta)^{|n'-n|} L^{(|n'-n|)}_{n_<}(eta^2)."""
    M = np.zeros((d, d), dtype=complex)
    for n in range(d):
        for m in range(d):
            lo, hi = min(n, m), max(n, m)
            M[m, n] = (
                np.exp(-(eta**2) / 2)
                * np.exp(0.5 * (gammaln(lo + 1) - gammaln(hi + 1)))
                * (1j * eta) ** (hi - lo)
                * eval_genlaguerre(lo, hi - lo, eta**2)
            )
    return M


try:
    import qutip as qt

    for d in (8, 10):
        cells, losses = [], []
        for eta in (0.1, 0.5, 1.0):
            De = qt.displace(d, 1j * eta).full()
            Da = analytic_D(d, eta)
            blk = d // 2
            cells.append(f"{np.abs(De[:blk, :blk] - Da[:blk, :blk]).max():.0e}")
            losses.append(f"{1 - np.sum(np.abs(Da[:, blk]) ** 2):.0e}")
        print(
            f"  d_m = {d:2d} | eta = 0.1 / 0.5 / 1.0 | max element difference, block n, n' < d_m/2: {' / '.join(cells)} | analytic norm loss, column d_m/2: {' / '.join(losses)}"
        )
except ImportError as e:
    print(f"  qutip not importable here: {e}")

# ---------------------------------------------------------------------------------------------
head("M. QuTiP 5.3.1 probes")
try:
    import qutip as qt

    print(f"  qutip {qt.__version__}")
    d = 8
    a = qt.destroy(d)
    gen1 = 1j * 0.08 * (a + a.dag())
    D1 = gen1.expm()
    print(
        f"  single-mode expm dtype: {D1.dtype.__name__}; nnz(dense view) = {np.count_nonzero(np.abs(D1.full()) > 0)} of {d * d}"
    )
    idm = qt.qeye(d)
    A1 = qt.tensor(qt.qeye(2), qt.qeye(2), a, idm, idm)
    A2 = qt.tensor(qt.qeye(2), qt.qeye(2), idm, a, idm)
    A3 = qt.tensor(qt.qeye(2), qt.qeye(2), idm, idm, a)
    gen = sum(
        1j * e * (A + A.dag()) for e, A in ((0.08, A1), (0.0824, A2), (0.047, A3))
    )
    Dj = gen.expm()
    print(
        f"  joint-generator expm dtype: {Dj.dtype.__name__} (the v3 benchmark's construction)"
    )
    Dprod = qt.tensor(
        qt.sigmap(),
        qt.qeye(2),
        *[(1j * e * (a + a.dag())).expm().to("CSR") for e in (0.08, 0.0824, 0.047)],
    )
    nnz = Dprod.data.as_scipy().nnz
    print(
        f"  sigma_+ (x) I (x) prod_m expm_m built per mode then tensored: dtype {Dprod.dtype.__name__}, nnz = {nnz}, formula 2^(N-1) prod d_m^2 = {2 * d**6}"
    )
    try:
        from qutip import enr_destroy, enr_fock

        ops = enr_destroy([7, 7], 6)
        Tq = qt.tensor(qt.sigmap(), ops[0])
        print(
            f"  ENR: tensor(sigmap, enr_destroy) dims = {Tq.dims}, shape = {Tq.shape} (product of dims = {np.prod(Tq.dims[0])})"
        )
        rho = qt.tensor(
            qt.fock_dm(2, 0),
            enr_fock([7, 7], 6, [0, 0]) * enr_fock([7, 7], 6, [0, 0]).dag(),
        )
        try:
            rho.ptrace(0)
            print("  ENR: ptrace succeeded")
        except Exception as e:  # report the limitation, do not hide it
            print(f"  ENR: ptrace raises {type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        print(f"  ENR probe failed: {type(e).__name__}: {e}")
    try:
        qt.mcsolve(
            qt.sigmaz(),
            qt.basis(2, 0),
            [0, 1.0],
            c_ops=[0.1 * qt.sigmam()],
            ntraj=10,
            target_tol=0.05,
        )
        print("  mcsolve(target_tol, no e_ops): ran")
    except Exception as e:
        print(
            f"  mcsolve(target_tol, no e_ops) raises {type(e).__name__}: {str(e)[:90]}"
        )
    H3 = qt.Qobj(np.diag([0.0, 1.0, 0.0]))
    H3 = H3 + 0.3 * (
        qt.basis(3, 0) * qt.basis(3, 1).dag() + qt.basis(3, 1) * qt.basis(3, 0).dag()
    )
    c3 = [
        np.sqrt(0.5) * qt.basis(3, 0) * qt.basis(3, 1).dag()
    ]  # |1> -> |0> only; |2> uncoupled
    for method in ("direct", "power", "eigen", "svd"):
        try:
            rs = qt.steadystate(H3, c3, method=method)
            print(
                f"  steadystate({method}) on a 3-level system with an uncoupled level: diag = {np.round(rs.diag().real, 4)} (kernel dimension 2: the method picks)"
            )
        except Exception as e:
            print(f"  steadystate({method}) raises {type(e).__name__}: {str(e)[:70]}")
except ImportError as e:
    print(f"  qutip not importable here: {e}")

print()
print("done.")
