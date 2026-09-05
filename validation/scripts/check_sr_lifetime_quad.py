"""Reduced quadrupole matrix element from the D5/2 lifetime.

PLAN Section 4.5.7 formula:
    A^(E2) = c_alpha * k^5 * |<S||r^2 C^(2)||D>|^2 / (15 (2 j' + 1)),  j' = 5/2
with c_alpha = e^2/(4 pi eps0 hbar) = 2.187691e6 m/s, and A = 1/tau at unit branching.
Inverted:  |<S||r^2 C^(2)||D>| = sqrt( 15 (2j'+1) A / (c_alpha k^5) ).
"""
import numpy as np

C_ALPHA = 2.187691e6          # m/s, e^2/(4 pi eps0 hbar)  (PLAN Sec 4.5.7)
A0 = 5.29177210903e-11        # m, Bohr radius (CODATA-2018)
A0_SQ = A0**2                 # m^2
JP = 2.5                      # upper level j'
DENOM = 15.0 * (2.0 * JP + 1.0)   # = 90 exactly


def reduced_element(lam_vac_m: float, tau_s: float, branching: float = 1.0):
    """Return (|<S||r^2 C2||D>| in m^2, same in a0^2, A in s^-1, k in m^-1)."""
    k = 2.0 * np.pi / lam_vac_m
    A = branching / tau_s
    theta_sq = DENOM * A / (C_ALPHA * k**5)   # m^4
    theta = np.sqrt(theta_sq)                 # m^2
    return theta, theta / A0_SQ, A, k


cases = [
    ("40Ca+  D5/2  (Barton 2000, PLAN check -> 9.73 a0^2)", 729.347e-9, 1.168),
    ("88Sr+  D5/2  Gerz 1987      tau = 0.345   s (PLAN current -> 14.70 a0^2)", 674.02559e-9, 0.345),
    ("88Sr+  D5/2  Madej/Sankey 1990 tau = 0.372 s", 674.02559e-9, 0.372),
    ("88Sr+  D5/2  Barwood 1994   tau = 0.347   s", 674.02559e-9, 0.347),
    ("88Sr+  D5/2  Biemont 2000   tau = 0.408   s", 674.02559e-9, 0.408),
    ("88Sr+  D5/2  Letchumanan thesis 2004 tau = 0.3903 s", 674.02559e-9, 0.3903),
    ("88Sr+  D5/2  Letchumanan PRA 2005 tau = 0.3908 s  <-- MODERN", 674.02559e-9, 0.3908),
]

print(f"{'case':<62} {'A /s^-1':>10} {'|<..>| /m^2':>14} {'/a0^2':>9}")
print("-" * 100)
for label, lam, tau in cases:
    theta, theta_a0, A, k = reduced_element(lam, tau)
    print(f"{label:<62} {A:10.6f} {theta:14.6e} {theta_a0:9.4f}")

print()
# uncertainty propagation for the modern value: element ~ sqrt(tau) => frac err = 0.5 * frac err(tau)
tau, dtau = 0.3908, 0.0016
th, th_a0, _, k = reduced_element(674.02559e-9, tau)
dth_a0 = 0.5 * (dtau / tau) * th_a0
print(f"88Sr+ modern:  tau = {tau*1e3:.1f}({dtau*1e3:.1f}) ms")
print(f"  A            = {1/tau:.6f} s^-1   (natural linewidth A/2pi = {1/tau/(2*np.pi):.4f} Hz)")
print(f"  k            = {k:.6e} m^-1  (lambda_vac = 674.02559 nm)")
print(f"  element      = {th:.6e} m^2 = {th_a0:.4f}({dth_a0:.4f}) a0^2")
print()
plan_now = 14.70
print(f"  PLAN current (tau = 0.345 s) : {plan_now:.2f} a0^2")
print(f"  change                       : {th_a0 - plan_now:+.4f} a0^2  ({100*(th_a0/plan_now - 1):+.2f} %)")
print(f"  ratio  = sqrt(0.345/0.3908)  = {np.sqrt(0.345/0.3908):.6f}  (element scales as 1/sqrt(tau))")
print()
# Rabi-frequency consequence: Omega ~ element, so same fractional change
print(f"  every 674 nm E2 Rabi frequency scales by the same {100*(th_a0/plan_now - 1):+.2f} %")
