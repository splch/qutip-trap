"""Langevin collision rate per ion for H2 background gas (Section 6.7).  Run: uv run --with scipy python check_collisions.py"""
import numpy as np
from scipy.constants import e, epsilon_0, k as kB, atomic_mass, pi, torr
alpha_vol = 0.80e-30            # H2 polarizability volume, m^3 (0.80 A^3)
alpha_si = 4 * pi * epsilon_0 * alpha_vol
for ion_u in (9.012, 40.08, 171.0):
    mu = 2.016 * ion_u / (2.016 + ion_u) * atomic_mass
    kL = (e / (2 * epsilon_0)) * np.sqrt(alpha_si / mu)        # m^3/s
    n = 1e-11 * torr / (kB * 300)                               # m^-3 at 1e-11 torr, 300 K
    G = n * kL
    print(f"ion {ion_u:6.1f} u: k_L = {kL*1e6:.2e} cm^3/s, n = {n*1e-6:.2e} cm^-3, Gamma_L = {G:.2e} s^-1 per ion, "
          f"one per {1/G/60:.0f} min per ion, one per {1/G/60/30:.1f} min for 30 ions")
