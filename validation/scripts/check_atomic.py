"""Self-checks for the atomic-structure layer the plan will specify.

1. Hyperfine + Zeeman Hamiltonian for a J = 1/2 ground manifold, diagonalized
   numerically in the |m_I, m_J> basis, compared with the closed-form
   Breit-Rabi expression, and used to locate field-independent (clock)
   transitions for 9Be+, 25Mg+, 43Ca+ and the 171Yb+ quadratic coefficient.
   Conventions (Steck): H = A_hfs I.J + mu_B (g_J J_z + g_I I_z) B / hbar-units,
   g_I = -(mu_I / (I mu_N)) (m_e / m_p)  [so g_I carries the electron-like sign].
2. Wigner-Eckart hyperfine dipole strengths for the 171Yb+ S1/2 F=1 -> P1/2 F'=0
   cycling line: relative line strengths and their sum rule.
3. Gamma -> reduced matrix element -> I_sat chain for 171Yb+ 369.5 nm and 40Ca+ 397 nm.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.constants import h, hbar, c, epsilon_0, e as qe, physical_constants
from sympy.physics.wigner import wigner_3j, wigner_6j

mu_B = physical_constants["Bohr magneton"][0]
mu_N = physical_constants["nuclear magneton"][0]
me_mp = physical_constants["electron mass"][0] / physical_constants["proton mass"][0]
mu_B_MHz_G = mu_B / h * 1e-6 * 1e-4  # MHz per gauss


def ang_ops(j):
    """Angular momentum matrices (units of hbar) in the |j, m> basis, m descending."""
    ms = np.arange(j, -j - 1, -1)
    dim = len(ms)
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        m = ms[i]
        Jp[i - 1, i] = np.sqrt(j * (j + 1) - m * (m + 1))
    Jm = Jp.conj().T
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2j)
    return Jx, Jy, Jz, ms


def hf_zeeman_levels(I, J, A_MHz, gJ, gI, B_G, B_MHz=0.0):
    """Eigenvalues (MHz) of A I.J + B_hfs term + mu_B (gJ Jz + gI Iz) B, and eigenvectors."""
    Ix, Iy, Iz, mI = ang_ops(I)
    Jx, Jy, Jz, mJ = ang_ops(J)
    dI, dJ = len(mI), len(mJ)
    kron = np.kron
    IdotJ = kron(Ix, Jx) + kron(Iy, Jy) + kron(Iz, Jz)
    H = A_MHz * IdotJ
    if B_MHz and I >= 1 and J >= 1:
        # electric quadrupole term (Steck convention)
        K = IdotJ
        H = H + B_MHz * (3 * K @ K + 1.5 * K - I * (I + 1) * J * (J + 1) * np.eye(dI * dJ)) / (
            2 * I * (2 * I - 1) * J * (2 * J - 1)
        )
    Hz = mu_B_MHz_G * B_G * (gJ * kron(np.eye(dI), Jz) + gI * kron(Iz, np.eye(dJ)))
    w, v = np.linalg.eigh(H + Hz)
    return w, v, mI, mJ


def breit_rabi(I, A_MHz, gJ, gI, B_G, F, mF):
    """Closed-form Breit-Rabi energy (MHz) for J = 1/2, F = I +- 1/2."""
    dE = A_MHz * (I + 0.5)  # hyperfine splitting
    x = (gJ - gI) * mu_B_MHz_G * B_G / dE
    sign = +1 if F == I + 0.5 else -1
    if abs(mF) == I + 0.5:  # stretched states: exact linear
        return dE * I / (2 * I + 1) + 0.5 * (gJ + 2 * I * gI) * mu_B_MHz_G * B_G * np.sign(mF)
    return (
        -dE / (2 * (2 * I + 1))
        + gI * mu_B_MHz_G * mF * B_G
        + sign * (dE / 2) * np.sqrt(1 + 4 * mF * x / (2 * I + 1) + x * x)
    )


def gI_from_mu(mu_I_in_muN, I):
    return -(mu_I_in_muN / I) * me_mp


def transition_freq(I, A, gJ, gI, B, Fa, ma, Fb, mb):
    return abs(breit_rabi(I, A, gJ, gI, B, Fa, ma) - breit_rabi(I, A, gJ, gI, B, Fb, mb))


def find_clock_point(I, A, gJ, gI, Fa, ma, Fb, mb, B_lo, B_hi):
    f = lambda B: transition_freq(I, A, gJ, gI, B, Fa, ma, Fb, mb)
    df = lambda B: (f(B + 1e-3) - f(B - 1e-3)) / 2e-3
    B0 = brentq(df, B_lo, B_hi)
    f0 = f(B0)
    curv = (f(B0 + 1e-2) - 2 * f0 + f(B0 - 1e-2)) / 1e-4  # MHz/G^2
    return B0, f0, curv


def check_breit_rabi_vs_numeric():
    print("\n=== 1a. Breit-Rabi closed form vs numerical diagonalization (J=1/2) ===")
    # 43Ca+: I = 7/2, A = -806.4020716 MHz, gJ = 2.00225664, mu_I = -1.31535 mu_N
    I, A, gJ, mu_I = 3.5, -806.4020716, 2.00225664, -1.31535
    gI = gI_from_mu(mu_I, I)
    for B in (0.0, 10.0, 146.0942, 500.0):
        w, v, mI, mJ = hf_zeeman_levels(I, 0.5, A, gJ, gI, B)
        br = sorted(breit_rabi(I, A, gJ, gI, B, F, mF) for F in (3, 4) for mF in range(-F, F + 1))
        print(
            f"43Ca+ B={B:9.4f} G  max|numeric - BreitRabi| = {np.max(np.abs(np.sort(w) - np.array(br))):.2e} MHz  (gI={gI:.6e})"
        )


def clock_points():
    print("\n=== 1b. Field-independent transitions (clock points) from the Breit-Rabi formula ===")
    # 43Ca+ 4,0 <-> 3,+1 : Harty 2014 gives 146.0942 G, 3 199 941 077 Hz, 2.4 mHz/mG^2
    I, A, gJ, gI = 3.5, -806.4020716, 2.00225664, gI_from_mu(-1.31535, 3.5)
    B0, f0, curv = find_clock_point(I, A, gJ, gI, 4, 0, 3, 1, 50, 300)
    print(
        f"43Ca+ |4,0>-|3,+1>: B0 = {B0:.4f} G, f0 = {f0 * 1e6:,.2f} Hz, d2f/dB2 = {curv * 1e6 * 1e-6:.3f} Hz/mG^2 -> {curv * 1e9 * 1e-6:.3f} mHz/mG^2   [Harty: 146.094 G, 3,199,941,077 Hz, 2.4 mHz/mG^2]"
    )
    # 9Be+ |2,0> <-> |1,1> : Langer 2005: 119.45 G, ~1.2075 GHz ; A = -625.008837048 MHz, mu_I = -1.177432 mu_N, gJ ~ 2.00226
    I, A, gJ, gI = 1.5, -625.008837048, 2.00226, gI_from_mu(-1.177432, 1.5)
    B0, f0, curv = find_clock_point(I, A, gJ, gI, 2, 0, 1, 1, 50, 300)
    print(
        f"9Be+  |2,0>-|1,+1>: B0 = {B0:.3f} G, f0 = {f0 * 1e6:,.1f} Hz, d2f/dB2 = {curv * 1e9 * 1e-6:.3f} mHz/mG^2   [Langer 2005: 119.45 G, ~1,207,495,843 Hz]"
    )
    # 25Mg+: A = -596.254376 MHz, I = 5/2, mu_I = -0.85545 mu_N, gJ ~ 2.00226; Srinivas 2021 use |3,1>-|2,1> at 212.8 G (1.686 GHz)
    I, A, gJ, gI = 2.5, -596.254376, 2.00226, gI_from_mu(-0.85545, 2.5)
    B0, f0, curv = find_clock_point(I, A, gJ, gI, 3, 1, 2, 1, 100, 400)
    print(
        f"25Mg+ |3,1>-|2,+1>: B0 = {B0:.2f} G, f0 = {f0 * 1e6:,.0f} Hz, d2f/dB2 = {curv * 1e9 * 1e-6:.3f} mHz/mG^2   [Srinivas 2021: 212.8 G, ~1.686 GHz]"
    )
    # 171Yb+ clock |0,0>-|1,0>: quadratic coefficient. ADOPTED constants (PLAN.md 4.5.1 after the 2026-09-04
    # critique): A = 12 642.8121185 MHz (Olmschenk 2007), I = 1/2, mu_I = +0.49367 mu_N, g_J = 2.002615
    # (Han et al. 2025). The retired g_J = 2.00254 gave 310.85 Hz/G^2 and the retired 2.00292 gave 310.78;
    # both are uncited and are kept ONLY as the labelled negative controls of check_critique_v3.py.
    I, A, gJ, gI = 0.5, 12642.8121185, 2.002615, gI_from_mu(0.49367, 0.5)
    f = lambda B: transition_freq(I, A, gJ, gI, B, 1, 0, 0, 0)
    coef = (f(1.0) - f(0.0)) * 1e6  # Hz per G^2 at 1 G: the Taylor c2 convention (Section 13)
    closed = ((gJ - gI) * mu_B_MHz_G) ** 2 / (2 * A) * 1e6
    print(
        f"171Yb+ clock quadratic shift: numeric {coef:.2f} Hz/G^2, closed form (gJ-gI)^2 mu_B^2/(2 h^2 A) = {closed:.2f} Hz/G^2   [PLAN.md 9.13: 310.87 +- 0.02 Hz/G^2]"
    )
    # 133Ba+ (I=1/2, A = 9925.45355459 MHz, Knab, Schupp and Werth, Europhys. Lett. 4, 1361 (1987)):
    # quadratic clock shift for reference, now with Ba+'s OWN measured g_J(6s 2S1/2) = 2.00249192(3) (Marx,
    # Tommaseo and Werth, Eur. Phys. J. D 4, 279 (1998), doi:10.1007/s100530050210; a Penning-trap
    # measurement on 138Ba+ and 135Ba+, isotope-independent to well inside 3e-8), which the tree carries as
    # ba133.S12.g_J. This script used to substitute 171Yb+'s 2.002615 and say no measured 133Ba+ g_J was
    # cited: it is cited, and the substitution biased the shift up by 1.2e-4 relative, 395.8 -> 395.7 Hz/G^2
    # at the printed precision.
    I, A, gJ = 0.5, 9925.45355459, 2.00249192
    closed = (gJ * mu_B_MHz_G) ** 2 / (2 * A) * 1e6
    print(f"133Ba+ clock quadratic shift, closed form ignoring gI: {closed:.1f} Hz/G^2")


def hyperfine_dipole_strengths():
    print("\n=== 2. Wigner-Eckart hyperfine line strengths (Steck convention) ===")

    # |<F mF| d_q |F' mF'>|^2 proportional to (2F'+1)(2F+1)(2J+1) * 3j^2 * 6j^2 with J, J' and I fixed.
    def strength(F, mF, Fp, mFp, q, J, Jp, I):
        w3 = float(wigner_3j(Fp, 1, F, mFp, q, -mF))
        w6 = float(wigner_6j(J, Jp, 1, Fp, F, I))
        return (2 * Fp + 1) * (2 * F + 1) * (2 * J + 1) * w3**2 * w6**2

    # 171Yb+: I = 1/2, S1/2 (J=1/2) -> P1/2 (J'=1/2). Transitions F=1 -> F'=0 (cycling) and F=1 -> F'=1, F=0 -> F'=1.
    I, J, Jp = 0.5, 0.5, 0.5
    tot = {}
    for F in (0, 1):
        for Fp in (0, 1):
            s = 0.0
            for mF in range(-F, F + 1):
                for q in (-1, 0, 1):
                    mFp = mF - q
                    if abs(mFp) <= Fp:
                        s += strength(F, mF, Fp, mFp, q, J, Jp, I)
            tot[(F, Fp)] = s
    print(
        "171Yb+ S1/2(F) -> P1/2(F') summed strengths S_FF' (normalized so that sum over F' for fixed F equals 1):"
    )
    for F in (0, 1):
        norm = sum(tot[(F, Fp)] for Fp in (0, 1))
        for Fp in (0, 1):
            print(f"   F={F} -> F'={Fp}: {tot[(F, Fp)] / norm:.6f}")
    # branching from P1/2 F'=0 back to S1/2 F=1 vs F=0
    for Fp in (0, 1):
        norm = sum(tot[(F, Fp)] for F in (0, 1))
        print(
            f"   decay branching P1/2 F'={Fp}: -> F=0 {tot[(0, Fp)] / norm:.4f}, -> F=1 {tot[(1, Fp)] / norm:.4f}   [expected F'=0 -> F=1 only (=1.0); F'=1 -> 1/3 F=0, 2/3 F=1]"
        )
    # 111Cd+/171Yb+ I=1/2 stretch-state cycling via P3/2: |1,1> -> |F'=2, 2> relative strength
    I, J, Jp = 0.5, 0.5, 1.5
    s_cycle = strength(
        1, 1, 2, 2, -1, J, Jp, I
    )  # sigma+ absorption: mF' = mF + 1 -> q = -1 in this convention
    s_all = sum(
        strength(1, 1, Fp, mFp, q, J, Jp, I)
        for Fp in (1, 2)
        for q in (-1, 0, 1)
        for mFp in [1 - q]
        if abs(mFp) <= Fp
    )
    print(
        f"   I=1/2 S1/2|1,1> -> P3/2: cycling |2,2> share of total absorption strength = {s_cycle / s_all:.4f}  [expect 1/2 with this sum over all q; sigma+ only: 1.0]"
    )


def isat_chain():
    print("\n=== 3. Gamma -> reduced dipole matrix element -> I_sat ===")
    # The total-versus-partial reading of PLAN.md 9.13's "quoted" 21.57 and 23.4 MHz is settled: both are
    # PARTIAL rates into S1/2, which is the reading this chain has always used and which the species table
    # now uses too. Hettrich et al., Phys. Rev. Lett. 115, 143003 (2015) print gamma_PS = 2 pi x 21.57(8) MHz
    # as the partial P1/2 -> S1/2 rate (with gamma_PD = 2 pi x 1.482(8) MHz and tau = 6.904(26) ns); as a
    # TOTAL rate 21.57 MHz would mean tau = 7.379 ns, which no measurement supports.
    # The table's own partial rates are the MEASURED lifetimes times the MEASURED S1/2 branchings:
    # tau(P1/2) = 6.904(26) ns (Hettrich) x 0.93565(7) (Ramm et al., Phys. Rev. Lett. 111, 023004 (2013))
    # and tau(P3/2) = 6.639(42) ns (Meir et al., Phys. Rev. A 101, 012509 (2020)) x (1 - 0.0587 - 0.00661)
    # (Gerritsma et al., Eur. Phys. J. D 50, 13 (2008)). The 397 nm one lands on the quoted 21.57 MHz to
    # 4e-5, so PLAN.md 9.13's 2.045 e a0 / 45.11 mW/cm^2 come out of the table; the 393 nm one is 22.407 MHz,
    # 4.2 % below the quoted 23.4, so the plan's 2.972 / 50.25 do not (ledger anchor.ca40.i_sat_chain,
    # anchor.ca40.p32_linewidth_readings).
    ca_p12_partial = 1e-6 / (2 * np.pi * 6.904e-9) * (1 - 0.06435)
    ca_p32_partial = 1e-6 / (2 * np.pi * 6.639e-9) * (1 - 0.0587 - 0.00661)
    for name, lam, Gamma_2pi_MHz, Jg, Je in (
        (
            "171Yb+ 369.5 nm S1/2-P1/2 (partial rate to S1/2: 19.72 MHz total x 0.99499)",
            369.52e-9,
            19.623,
            0.5,
            0.5,
        ),
        # Section 13 and 4.5.7: lambda is the VACUUM wavelength. The 396.85 / 393.37 nm this script used to
        # take are the AIR values; PLAN.md 9.13's 2.045 e a0 / 45.11 and 2.972 / 50.25 were computed at them,
        # so both wavelengths are printed for each line.
        (
            "40Ca+ 397 nm S1/2-P1/2 (vacuum 396.959 nm, quoted 21.57 MHz as the partial rate)",
            396.959e-9,
            21.57,
            0.5,
            0.5,
        ),
        ("40Ca+ 397 nm S1/2-P1/2 (PLAN.md 9.13 input: air 396.85 nm)", 396.85e-9, 21.57, 0.5, 0.5),
        (
            "40Ca+ 397 nm S1/2-P1/2 (table: Hettrich tau x Ramm branching, vacuum)",
            396.959e-9,
            ca_p12_partial,
            0.5,
            0.5,
        ),
        (
            "40Ca+ 393 nm S1/2-P3/2 (vacuum 393.478 nm, quoted 23.4 MHz as the partial rate)",
            393.478e-9,
            23.4,
            0.5,
            1.5,
        ),
        ("40Ca+ 393 nm S1/2-P3/2 (PLAN.md 9.13 input: air 393.37 nm)", 393.37e-9, 23.4, 0.5, 1.5),
        (
            "40Ca+ 393 nm S1/2-P3/2 (table: Meir tau x Gerritsma branching, vacuum)",
            393.478e-9,
            ca_p32_partial,
            0.5,
            1.5,
        ),
    ):
        Gamma = 2 * np.pi * Gamma_2pi_MHz * 1e6
        omega = 2 * np.pi * c / lam
        # Steck: Gamma = omega^3/(3 pi eps0 hbar c^3) * (2Jg+1)/(2Je+1) * |<Jg||d||Je>|^2
        d2 = Gamma * 3 * np.pi * epsilon_0 * hbar * c**3 / omega**3 * (2 * Je + 1) / (2 * Jg + 1)
        d = np.sqrt(d2)
        Isat = np.pi * h * c * Gamma / (3 * lam**3)  # two-level cycling I_sat (Steck), W/m^2
        # Rabi frequency at I = I_sat for the cycling (stretched) transition: Omega = Gamma sqrt(I/(2 I_sat))
        Om = Gamma * np.sqrt(0.5)
        print(
            f"{name}: |<J||d||J'>| = {d / (qe * physical_constants['Bohr radius'][0]):.3f} e a0, I_sat = {Isat * 0.1:.2f} mW/cm^2, Omega(I=I_sat)/2pi = {Om / 2 / np.pi / 1e6:.2f} MHz"
        )


if __name__ == "__main__":
    check_breit_rabi_vs_numeric()
    clock_points()
    hyperfine_dipole_strengths()
    isat_chain()
