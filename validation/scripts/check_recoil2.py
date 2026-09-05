"""Second recoil check: numbers quoted from the recoil follow-up brief (Run 4R), recomputed locally.

Run: uv run --with qutip --with scipy python check_recoil2.py
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import hbar, atomic_mass, c, k as kB, pi
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, factorial

print(
    "== 40Ca+ Lamb-Dicke parameters at 2pi x 1 MHz (Roos thesis p. 28: 0.096 and 0.179) =="
)
M = 40 * atomic_mass  # Roos uses 40 u
nu = 2 * pi * 1e6
x0 = np.sqrt(hbar / (2 * M * nu))
for lam in (729.147e-9, 393.366e-9):
    k = 2 * pi / lam
    print(f"  lambda = {lam * 1e9:.3f} nm  eta = {k * x0:.4f}")
print(f"  x0 = {x0 * 1e9:.3f} nm")
print(
    f"  (eta_393/eta_729)^2 = {(729.147 / 393.366) ** 2:.3f}; alpha=2/5 -> (eta_t/eta)^2 = {0.4 * (729.147 / 393.366) ** 2:.3f}; bracket = {0.4 * (729.147 / 393.366) ** 2 + 0.25:.3f}"
)

print("\n== Quadrature identities: Gauss-Legendre M=16 on [-1,1] ==")
u, w = leggauss(16)
for name, ft, alpha_exp in (
    ("pi (mode along B)", lambda u: 0.75 * (1 - u**2), 0.2),
    ("sigma (mode along B)", lambda u: 0.375 * (1 + u**2), 0.4),
    ("isotropic", lambda u: 0.5 * np.ones_like(u), 1 / 3),
):
    p = w * ft(u)
    print(
        f"  {name:22s} sum p = {p.sum():.15f}  sum p u^2 = {(p * u**2).sum():.15f}  (alpha {alpha_exp})"
    )

print(
    "\n== Single-kick test: Kraus map once on |e,0>, eta_em = 0.1, sigma pattern -> <n> = alpha eta^2 = 0.004 =="
)
N = 40
a = np.diag(np.sqrt(np.arange(1, N)), 1)
x = a + a.T
eta = 0.1
p = w * 0.375 * (1 + u**2)
psi0 = np.zeros(N)
psi0[0] = 1
rho = np.zeros((N, N), dtype=complex)
total = 0.0
for pj, uj in zip(p, u):
    D = expm(-1j * eta * uj * x)
    v = D @ psi0
    rho += pj * np.outer(v, v.conj())
    total += pj
n_op = np.diag(np.arange(N))
p_op = 1j * (a.T - a)
print(
    f"  trace = {np.trace(rho).real:.15f}  <n> = {np.trace(n_op @ rho).real:.10f}  <p>/p0 = {np.trace(p_op @ rho).real:.2e}"
)

print(
    "\n== Resolved-sideband floor: n = (G/2nu)^2 [alpha/cos^2 + 1/4]; alpha -> 0 at G/nu = 0.01 gives (G/4nu)^2 =="
)
G_over_nu = 0.01
print(f"  alpha = 0: {(G_over_nu / 4) ** 2:.6e}")
print(
    f"  alpha = 0.4: {(G_over_nu / 2) ** 2 * (0.4 + 0.25):.6e}; recoil share = {0.4 / (0.4 + 0.25):.3f}"
)

print(
    "\n== Eschner Eq. 6 rate coefficients: Omega cancels from n_ss; Doppler optimum at Delta = -Gamma/2 =="
)


def A(Delta, nu, G, alpha, cos2=1.0, eta=0.05, Om=0.01):
    W = lambda d: 1 / (4 * d**2 / G**2 + 1)
    Ap = (Om**2 / G) * eta**2 * (cos2 * W(Delta - nu) + alpha * W(Delta))
    Am = (Om**2 / G) * eta**2 * (cos2 * W(Delta + nu) + alpha * W(Delta))
    return Ap, Am


for Om in (0.01, 0.02):
    Ap, Am = A(-1.0, 1.0, 0.01, 0.4, Om=Om)
    print(f"  Omega = {Om}: n_ss = {Ap / (Am - Ap):.6e}")
# Doppler regime: nu << Gamma; scan Delta
G = 1.0
nu_d = 1e-3
Ds = np.linspace(-2, -0.05, 20001)
nss = []
for D in Ds:
    Ap, Am = A(D, nu_d, G, 0.4)
    nss.append(Ap / (Am - Ap))
nss = np.array(nss)
i = np.argmin(nss)
print(
    f"  Doppler: argmin Delta/Gamma = {Ds[i]:.4f}; n_min = {nss[i]:.4f}; (G/4nu)(1+alpha) = {(G / (4 * nu_d)) * (1.4):.4f}"
)
print(
    f"  Itano E_K/(hbar Gamma) = (1+alpha)/8: alpha=1/3 -> {(1 + 1 / 3) / 8:.5f} (1/6 = {1 / 6:.5f}); alpha=2/5 -> {1.4 / 8:.5f} (7/40 = {7 / 40:.5f})"
)

print(
    "\n== Cirac Fig. 8 minima: <n>_min = 1/sqrt(beta) - 1/2 at Omega_phi^2 = 4 Delta nu / sqrt(beta) (Gamma=1, nu=0.1, Delta=10) =="
)
for beta in (0.9, 0.1):
    Om2 = 4 * 10 * 0.1 / np.sqrt(beta)
    E = 0.5 * 0.1 * (Om2 / (4 * 10 * 0.1) + 4 * 10 * 0.1 / (beta * Om2))
    n = E / 0.1 - 0.5
    print(
        f"  beta = {beta}: Omega_phi = {np.sqrt(Om2):.3f}, <n>_min = {n:.3f} (1/sqrt(beta)-1/2 = {1 / np.sqrt(beta) - 0.5:.3f})"
    )
print(f"  Doppler reference Gamma/(2nu) - 1/2 = {1 / (2 * 0.1) - 0.5:.1f}")

print(
    "\n== Franck-Condon: exact expm vs Roos Eq. 3.11 as printed (L_n) vs corrected (L_{n_<}) at eta = 0.05 =="
)
N = 200
a = np.diag(np.sqrt(np.arange(1, N)), 1)
eta = 0.05
Dm = expm(1j * eta * (a + a.T))
for n, m in ((1, -1), (3, -2), (8, -3), (2, 2)):
    exact = abs(Dm[n + m, n])
    nlo, nhi = min(n, n + m), max(n, n + m)
    corrected = (
        np.exp(-(eta**2) / 2)
        * eta ** abs(m)
        * abs(eval_genlaguerre(nlo, abs(m), eta**2))
        * np.sqrt(factorial(nlo) / factorial(nhi))
    )
    printed = (
        np.exp(-(eta**2) / 2)
        * eta ** abs(m)
        * abs(eval_genlaguerre(n, abs(m), eta**2))
        * np.sqrt(factorial(nlo) / factorial(nhi))
    )
    print(
        f"  (n,m)=({n},{m}): exact {exact:.8f}  corrected {corrected:.8f}  printed {printed:.8f}  printed/exact = {printed / exact:.2f}"
    )

print(
    "\n== Cooling-rate saturation R_n = G (eta sqrt(n) Om)^2 / [2 (eta sqrt(n) Om)^2 + G^2] =="
)
G = 1.0
for x_ in (0.01, 0.1, 1.0, 10.0, 100.0):
    print(
        f"  eta sqrt(n) Om / G = {x_:6.2f}: R_n/G = {x_**2 / (2 * x_**2 + 1):.6f}  weak-drive (eta sqrt n Om)^2/G = {x_**2:.4f}"
    )

print("\n== Effective two-level after adiabatic elimination: gamma'/Gamma' ==")
G10, G12 = 1.0, 0.1
print(
    f"  Xi: gamma'/Gamma' = (G10+G12)/(2 G10) = {(G10 + G12) / (2 * G10):.3f};  V: (G10+G12)/(2 G12) = {(G10 + G12) / (2 * G12):.3f}"
)
# Marzoli Fig. 3 light shift: delta' = delta_20 - delta_12 * Pi_12, Pi = (Om/2)^2/(delta^2 + ((G10+G12)/2)^2)
Om12, d12 = 0.3, -1.0
Pi12 = (Om12 / 2) ** 2 / (d12**2 + ((G10 + G12) / 2) ** 2)
print(
    f"  Marzoli Fig. 3 (Om12 = 0.3, d12 = -1): light shift -d12 Pi12 = {-d12 * Pi12:+.4f} G10; optimum delta_20 = -nu - shift = {-0.05 - (-d12 * Pi12):+.4f} G10 (brief: -0.058)"
)

print(
    "\n== Neutral-alkali recoil regression (Steck): v_r = hbar k / m, k_B T_r = (hbar k)^2 / m =="
)
for name, mass_u, lam in (("133Cs", 132.905, 852.347e-9), ("87Rb", 86.909, 780.241e-9)):
    m = mass_u * atomic_mass
    k = 2 * pi / lam
    print(
        f"  {name}: v_r = {hbar * k / m * 1e3:.2f} mm/s, T_r = {(hbar * k) ** 2 / (m * kB) * 1e9:.1f} nK"
    )

print("\n== Itano recoil-vs-linewidth ratios ==")
for name, mass_u, lam, gamma_hz in (
    ("138Ba+", 137.905, 493.4e-9, 21e6),
    ("24Mg+", 23.985, 279.6e-9, 43e6),
):
    m = mass_u * atomic_mass
    k = 2 * pi / lam
    R = (hbar * k) ** 2 / (2 * m)
    print(
        f"  {name}: R/h = {R / (2 * pi * hbar) / 1e3:.1f} kHz; R/(hbar gamma) = {R / (hbar * 2 * pi * gamma_hz):.2e}"
    )
