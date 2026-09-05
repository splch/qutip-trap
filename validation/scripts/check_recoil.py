"""Recoil angular factors alpha = <(k_hat . e_mode)^2> over the dipole emission patterns,
and the 171Yb+ recoil-per-photon numbers quoted in Section 4.2.8."""

import numpy as np
from scipy.integrate import dblquad
from scipy.constants import hbar, atomic_mass, pi


def alpha(pattern, axis):
    # pattern: N(theta) normalized over the sphere, theta measured from the quantization axis z
    if pattern == "pi":
        N = lambda th: 3 / (8 * pi) * np.sin(th) ** 2
    elif pattern == "sigma":
        N = lambda th: 3 / (16 * pi) * (1 + np.cos(th) ** 2)
    else:
        N = lambda th: 1 / (4 * pi)
    if axis == "z":
        proj = lambda th, ph: np.cos(th) ** 2
    else:  # x axis, perpendicular to B
        proj = lambda th, ph: (np.sin(th) * np.cos(ph)) ** 2
    norm = dblquad(lambda th, ph: N(th) * np.sin(th), 0, 2 * pi, 0, pi)[0]
    val = dblquad(lambda th, ph: N(th) * proj(th, ph) * np.sin(th), 0, 2 * pi, 0, pi)[0]
    return val, norm


for pattern in ("pi", "sigma", "iso"):
    for axis in ("z", "x"):
        v, n = alpha(pattern, axis)
        print(
            f"alpha[{pattern:5s}, axis {axis} ({'along B' if axis == 'z' else 'perp to B'})] = {v:.6f}  (pattern norm {n:.6f})"
        )

m = 171 * atomic_mass
for lam, omega_2pi, label in (
    (369.5e-9, 3e6, "369.5 nm detection/cooling photon, 3 MHz mode"),
    (355e-9, 3e6, "355 nm Raman photon, 3 MHz mode"),
):
    omega = 2 * pi * omega_2pi
    x0 = np.sqrt(hbar / (2 * m * omega))
    k = 2 * pi / lam
    eta = k * x0
    print(
        f"171Yb+, {label}: x0 = {x0 * 1e9:.3f} nm, k x0 = {eta:.4f}, (k x0)^2 = {eta**2:.3e}, recoil quanta per photon at alpha=2/5: {0.4 * eta**2:.2e}; x1e4 photons: {0.4 * eta**2 * 1e4:.1f}"
    )
Om = 2 * pi * 1e6
print(
    f"counter-rotating Raman shift |Omega|^2/(2 omega0) for Omega/2pi = 1 MHz, omega0/2pi = 12.6428 GHz: {Om**2 / (2 * 2 * pi * 12.6428e9) / (2 * pi):.1f} Hz"
)
