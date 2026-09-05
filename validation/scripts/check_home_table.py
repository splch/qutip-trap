"""Home 2013 Table I (Be+/Mg+ two-ion crystal): which assignment of the 24Mg+ single-ion x, y frequencies
reproduces the printed off-species amplitudes 0.018 (x) and 0.020 (y)?

Run: uv run --with numpy --with scipy python check_home_table.py
"""

import numpy as np
from scipy.constants import e, epsilon_0, atomic_mass, pi

mBe, mMg = 9.0121822 * atomic_mass, 23.985042 * atomic_mass
wBe = 2 * pi * np.array([12.26, 11.19, 2.69]) * 1e6
C = e**2 / (4 * pi * epsilon_0)


def modes(wMg):
    kz = mBe * wBe[2] ** 2  # dc axial curvature, mass independent
    z = (C / (4 * kz)) ** (1 / 3)  # half separation: kz z = C/(2z)^2
    out = {}
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        kBe, kMg = mBe * wBe[i] ** 2, mMg * wMg[i] ** 2
        c = 2 * C / (2 * z) ** 3 if ax == "z" else -C / (2 * z) ** 3
        K = np.array([[kBe + c, -c], [-c, kMg + c]])
        Minv = np.diag([1 / np.sqrt(mBe), 1 / np.sqrt(mMg)])
        w2, v = np.linalg.eigh(Minv @ K @ Minv)
        out[ax] = (np.sqrt(w2) / (2 * pi * 1e6), v)
    return out, z


print(
    f"Mg axial single-ion from dc scaling: {2.69 * np.sqrt(mBe / mMg):.3f} MHz (table 1.65)"
)
for label, wMg in (
    ("printed caption (3.72, 4.82, 1.65)", [3.72, 4.82, 1.65]),
    ("corrected (4.82, 3.72, 1.65)", [4.82, 3.72, 1.65]),
):
    out, z = modes(2 * pi * np.array(wMg) * 1e6)
    print(f"{label}   separation {2 * z * 1e6:.2f} um")
    for ax in "xyz":
        f, v = out[ax]
        hi, lo = v[:, 1], v[:, 0]
        print(
            f"   {ax}: {f[1]:.3f} MHz (Be {abs(hi[0]):.3f}, Mg {abs(hi[1]):.3f}); "
            f"{f[0]:.3f} MHz (Be {abs(lo[0]):.3f}, Mg {abs(lo[1]):.3f})"
        )
