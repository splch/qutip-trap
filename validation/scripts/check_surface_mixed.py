"""Checks for the surface-trap and mixed-species formulas to be written into the plan.

S1: gapless-plane potential of a rectangular electrode (solid-angle form): Laplace equation,
    boundary values, and the infinite-strip limit.
S2: symmetric five-wire trap: rf null height h = sqrt(a (a + b)) for centre half-width a and
    rf strip width b, from the strip potential; pseudopotential curvature there.
M1: mixed-species two-ion axial modes from the mass-weighted Hessian against the closed form
    w^2/wz1^2 = 1 + 1/mu +- sqrt(1 - 1/mu + 1/mu^2); eigenvector normalization and eta.
"""

import numpy as np
from scipy.optimize import brentq


def phi_rect(x, y, z, x1, x2, y1, y2):
    """Potential of a rectangle [x1,x2]x[y1,y2] at unit voltage in the z=0 gapless plane (solid angle / 2 pi)."""
    tot = 0.0
    for i, xi in enumerate((x1, x2)):
        for j, yj in enumerate((y1, y2)):
            R = np.sqrt((xi - x) ** 2 + (yj - y) ** 2 + z**2)
            tot += (-1) ** (i + j) * np.arctan((xi - x) * (yj - y) / (z * R))
    return tot / (2 * np.pi)


def laplacian(f, x, y, z, h=1e-4):
    return (
        f(x + h, y, z)
        + f(x - h, y, z)
        + f(x, y + h, z)
        + f(x, y - h, z)
        + f(x, y, z + h)
        + f(x, y, z - h)
        - 6 * f(x, y, z)
    ) / h**2


def S1():
    print("=== S1: rectangular electrode, gapless plane ===")
    f = lambda x, y, z: phi_rect(x, y, z, -1.0, 1.0, -0.5, 0.5)
    for pt in ((0.2, 0.1, 0.7), (1.5, 0.3, 0.4), (0.0, 0.0, 2.0)):
        print(f"  Laplacian at {pt}: {laplacian(f, *pt):.2e} (potential {f(*pt):.5f})")
    print(
        f"  boundary: inside electrode z->0: {f(0.3, 0.2, 1e-6):.6f} (expect 1); outside: {f(1.5, 0.0, 1e-6):.2e} (expect 0)"
    )
    # strip limit: y1, y2 -> +-inf compared with (1/pi)[atan((x2-x)/z) - atan((x1-x)/z)]
    strip = lambda x, z, x1, x2: (
        (np.arctan((x2 - x) / z) - np.arctan((x1 - x) / z)) / np.pi
    )
    print(
        f"  strip limit: rect(y +-1e6) {phi_rect(0.3, 0.0, 0.8, -1, 1, -1e6, 1e6):.8f} vs strip {strip(0.3, 0.8, -1, 1):.8f}"
    )
    # solid angle of a full plane -> 2 pi -> potential 1 far above a huge electrode
    print(
        f"  huge electrode, z=1: {phi_rect(0, 0, 1.0, -1e6, 1e6, -1e6, 1e6):.8f} (expect 1)"
    )


def S2():
    print("\n=== S2: symmetric five-wire trap rf null height ===")
    a, b = (
        0.5,
        1.2,
    )  # centre electrode half-width a, rf strips from a to a+b on both sides

    def Ez_axis(z):
        # E_z = -d/dz of the strip potentials at unit voltage on both rf strips
        def dphi_dz(x1, x2):
            # d/dz (1/pi)[atan((x2-x)/z) - atan((x1-x)/z)] at x = 0
            return (1 / np.pi) * (-(x2) / (z**2 + x2**2) + (x1) / (z**2 + x1**2))

        return -(dphi_dz(a, a + b) + dphi_dz(-(a + b), -a))

    h_num = brentq(Ez_axis, 1e-3, 10)
    print(
        f"  a={a}, b={b}: numerical null height {h_num:.6f}, sqrt(a(a+b)) = {np.sqrt(a * (a + b)):.6f}"
    )

    # pseudopotential curvature at the null: Psi = Q^2 |E|^2/(4 m Omega^2); compute |E|^2 Hessian numerically (unit V)
    def E2(x, z):
        hh = 1e-6

        def phi(xx, zz):
            s = lambda x1, x2: (
                (np.arctan((x2 - xx) / zz) - np.arctan((x1 - xx) / zz)) / np.pi
            )
            return s(a, a + b) + s(-(a + b), -a)

        Ex = -(phi(x + hh, z) - phi(x - hh, z)) / (2 * hh)
        Ez = -(phi(x, z + hh) - phi(x, z - hh)) / (2 * hh)
        return Ex**2 + Ez**2

    hh = 1e-3
    cxx = (E2(hh, h_num) - 2 * E2(0, h_num) + E2(-hh, h_num)) / hh**2
    czz = (E2(0, h_num + hh) - 2 * E2(0, h_num) + E2(0, h_num - hh)) / hh**2
    print(
        f"  d2|E|^2/dx2 = {cxx:.5f}, d2|E|^2/dz2 = {czz:.5f} (per unit V^2 and length^-4); ratio z/x = {czz / cxx:.4f} (expect 1: isotropic rf curvature at the null of a symmetric five-wire trap)"
    )


def M1():
    print("\n=== M1: mixed-species two-ion axial modes ===")
    m1, m2 = 9.012, 24.305  # 9Be+, 24Mg+ (amu units, ratios only)
    mu = m2 / m1
    # dimensionless: single-ion axial frequency of ion 1 = 1 (potential 1/2 m1 wz1^2 x^2 for both ions since the trap is a potential on charge)
    # V = 1/2 m1 wz1^2 (x1^2 + x2^2) + kC/|x1-x2| with kC chosen so that the equilibrium separation is d: kC = m1 wz1^2 d^3 / 2 ... use d = 2^(1/3) l with l^3 = kC/(m1 wz1^2)
    m1w2 = 1.0
    kC = 1.0  # sets length scale l = 1
    d = 2 ** (1 / 3)
    # Hessian of V at equilibrium (+-d/2)
    K = np.array(
        [[m1w2 + 2 * kC / d**3, -2 * kC / d**3], [-2 * kC / d**3, m1w2 + 2 * kC / d**3]]
    )
    M = np.diag([m1, m2]) / m1  # in units of m1; then wz1^2 = m1w2/m1 -> 1
    Minv_half = np.diag(1 / np.sqrt(np.diag(M)))
    w2, C = np.linalg.eigh(Minv_half @ K @ Minv_half)
    closed = np.array(
        [
            1 + 1 / mu - np.sqrt(1 - 1 / mu + 1 / mu**2),
            1 + 1 / mu + np.sqrt(1 - 1 / mu + 1 / mu**2),
        ]
    )
    print(f"  mu = {mu:.4f}: numerical w^2/wz1^2 = {w2}, closed form = {closed}")
    # eigenvector normalization: c orthonormal; b_i = c_i/sqrt(m_i) ; check x_i expansion normalization sum_k m_i b_ik^2 = 1
    b = Minv_half @ C
    print(
        f"  orthonormality of c: {np.abs(C.T @ C - np.eye(2)).max():.1e}; sum_k m_i b_ik^2 = {np.sum(np.diag(M)[:, None] * b**2, axis=1)} (expect [1, 1])"
    )
    # equal-mass check: mu=1 -> w^2 = 1, 3
    M1 = np.eye(2)
    w2e, _ = np.linalg.eigh(K)
    print(f"  equal masses: w^2/wz^2 = {w2e} (expect [1, 3])")


if __name__ == "__main__":
    S1()
    S2()
    M1()
