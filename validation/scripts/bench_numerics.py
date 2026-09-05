"""Numerical checks for the plan's Section 5 and 11 claims.

B1: stiffness of the motion-Schroedinger picture vs the interaction picture
    (sideband decomposition, all sidebands retained) for a resonant carrier
    pulse and for a detuned (MS-like) spin-dependent-force pulse.
B2: displacement operator in a truncated Fock space: expm of the truncated
    generator vs the analytic (Laguerre) matrix elements; unitarity defect;
    ENR-space expm vs analytic elements.
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.special import gammaln, eval_genlaguerre
import qutip as q

np.set_printoptions(precision=4, linewidth=140)


def analytic_displacement(nmax, alpha):
    """<n'|D(alpha)|n> for n, n' < nmax, exact infinite-space elements projected."""
    N = nmax
    D = np.zeros((N, N), dtype=complex)
    x = abs(alpha) ** 2
    for n in range(N):
        for m in range(N):
            # <m|D|n>; k = m - n
            if m >= n:
                k = m - n
                pref = np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))
                D[m, n] = pref * alpha**k * np.exp(-x / 2) * eval_genlaguerre(n, k, x)
            else:
                k = n - m
                pref = np.exp(0.5 * (gammaln(m + 1) - gammaln(n + 1)))
                D[m, n] = (
                    pref
                    * (-np.conj(alpha)) ** k
                    * np.exp(-x / 2)
                    * eval_genlaguerre(m, k, x)
                )
    return D


def sideband_parts(D):
    N = D.shape[0]
    parts = {}
    for k in range(-(N - 1), N):
        A = np.zeros_like(D)
        for n in range(N):
            m = n + k
            if 0 <= m < N:
                A[m, n] = D[m, n]
        if np.any(A):
            parts[k] = A
    return parts


def run_ivp(Hfun, psi0, T, rtol=1e-8, atol=1e-10):
    def rhs(t, y):
        return -1j * (Hfun(t) @ y)

    t0 = time.perf_counter()
    sol = solve_ivp(
        rhs, (0.0, T), psi0, method="DOP853", rtol=rtol, atol=atol, t_eval=[T]
    )
    dt = time.perf_counter() - t0
    return sol.y[:, -1], sol.nfev, dt


def bench_B1():
    print(
        "\n=== B1: stiffness, Schroedinger-motion vs interaction picture (solve_ivp DOP853, rtol 1e-8, atol 1e-10) ==="
    )
    omega = 2 * np.pi * 3.0e6
    eta = 0.1
    Om = 2 * np.pi * 250e3  # carrier Rabi frequency, pi time 2 us
    cases = [
        ("carrier, delta=0,      n0=1, T=2us", 0.0, 1, 2e-6),
        ("carrier, delta=10kHz,  n0=1, T=2us", 2 * np.pi * 10e3, 1, 2e-6),
        (
            "SDF: delta=omega-20kHz (blue sb detuned), n0=0, T=10us",
            omega - 2 * np.pi * 20e3,
            0,
            10e-6,
        ),
    ]
    sm = np.array(
        [[0, 0], [1, 0]], dtype=complex
    )  # sigma+ : |down>=e1 -> |up>=e0  (QuTiP: sigmap raises basis(2,1)->basis(2,0))
    for label, delta, n0, T in cases:
        print(f"\n-- {label}")
        for nmax in (6, 11, 21, 41):
            D = analytic_displacement(nmax, 1j * eta)
            a = np.diag(np.sqrt(np.arange(1, nmax)), 1)
            H0 = np.kron(np.eye(2), omega * a.conj().T @ a)
            V = np.kron(sm, D)  # sigma+ (x) D
            Vd = V.conj().T
            H0s, Vs, Vds = (sp.csr_matrix(M) for M in (H0, V, Vd))

            # Schroedinger picture
            def HS(t):
                c = 0.5 * Om * np.exp(-1j * delta * t)
                return H0s + c * Vs + np.conj(c) * Vds

            # interaction picture: sideband decomposition
            parts = sideband_parts(D)
            Ak = {k: sp.csr_matrix(np.kron(sm, A)) for k, A in parts.items()}
            Akd = {k: M.conj().T.tocsr() for k, M in Ak.items()}

            def HI(t):
                c = 0.5 * Om * np.exp(-1j * delta * t)
                H = None
                for k, M in Ak.items():
                    ck = c * np.exp(1j * k * omega * t)
                    term = ck * M + np.conj(ck) * Akd[k]
                    H = term if H is None else H + term
                return H

            psi0 = np.zeros(2 * nmax, dtype=complex)
            psi0[nmax + n0] = 1.0  # |down, n0>
            psiS, nS, tS = run_ivp(HS, psi0, T)
            psiI, nI, tI = run_ivp(HI, psi0, T)
            # back to Schroedinger picture: psi_S = exp(-i H0 T) psi_I
            phase = np.exp(-1j * np.diag(H0) * T)
            psiI_S = phase * psiI
            diff = np.linalg.norm(psiS - psiI_S)
            pup = np.sum(np.abs(psiS[:nmax]) ** 2)
            print(
                f"nmax={nmax:3d} dim={2 * nmax:4d} | S: nfev={nS:7d} {tS:6.2f}s | I({len(Ak)} sb terms): nfev={nI:7d} {tI:6.2f}s | "
                f"|psiS-psiI|={diff:.1e}  P_up={pup:.6f}  nfev ratio S/I={nS / nI:.1f}"
            )


def bench_B1_qutip():
    print(
        "\n=== B1b: QuTiP sesolve wall time, motion-Schroedinger picture, carrier pulse (vern9 / adams / dop853) ==="
    )
    omega = 2 * np.pi * 3.0e6
    eta = 0.1
    Om = 2 * np.pi * 250e3
    delta = 2 * np.pi * 10e3
    T = 2e-6
    for nmax in (11, 21, 41):
        D = q.Qobj(analytic_displacement(nmax, 1j * eta))
        a = q.destroy(nmax)
        H0 = q.tensor(q.qeye(2), omega * a.dag() * a)
        V = q.tensor(q.sigmap(), D)
        H = q.QobjEvo(
            [
                H0,
                [V, lambda t: 0.5 * Om * np.exp(-1j * delta * t)],
                [V.dag(), lambda t: 0.5 * Om * np.exp(1j * delta * t)],
            ]
        )
        psi0 = q.tensor(q.basis(2, 1), q.basis(nmax, 1))
        for method in ("vern9", "adams", "dop853"):
            t0 = time.perf_counter()
            res = q.sesolve(
                H,
                psi0,
                [0, T],
                options={
                    "method": method,
                    "atol": 1e-10,
                    "rtol": 1e-8,
                    "store_final_state": True,
                },
            )
            dt = time.perf_counter() - t0
            stats = {
                k: v
                for k, v in res.stats.items()
                if k in ("num_steps", "run time", "method")
            }
            print(f"nmax={nmax:3d} {method:7s} wall={dt:6.2f}s stats={stats}")
    print("available stats keys:", list(res.stats.keys()))


def bench_B2():
    print("\n=== B2: displacement operator in truncated Fock space ===")
    for nmax in (10, 20, 40):
        for eta in (0.1, 0.5, 1.0):
            Dex = analytic_displacement(nmax, 1j * eta)
            Dexpm = q.displace(nmax, 1j * eta).full()
            half = nmax // 2
            d_low = np.max(np.abs((Dex - Dexpm)[:half, :half]))
            d_all = np.max(np.abs(Dex - Dexpm))
            I = np.eye(nmax)
            u_ex = np.linalg.norm(Dex.conj().T @ Dex - I, 2)
            u_expm = np.linalg.norm(Dexpm.conj().T @ Dexpm - I, 2)
            # Rabi matrix elements of interest
            om00 = abs(Dex[0, 0])
            om00e = abs(Dexpm[0, 0])
            omb = abs(Dex[nmax - 1, nmax - 2])
            ombe = abs(Dexpm[nmax - 1, nmax - 2])
            # column norm loss of analytic D at the boundary column (probability leaving the truncated space)
            loss_top = 1 - np.sum(np.abs(Dex[:, nmax - 1]) ** 2)
            loss_mid = 1 - np.sum(np.abs(Dex[:, half]) ** 2)
            print(
                f"nmax={nmax:3d} eta={eta:.1f} | max|Dex-Dexpm| low-block={d_low:.1e} all={d_all:.1e} | "
                f"unitarity defect: analytic={u_ex:.1e} expm={u_expm:.1e} | Omega00 an={om00:.10f} expm={om00e:.10f} | "
                f"boundary element an={omb:.4f} expm={ombe:.4f} | analytic norm loss: col nmax/2={loss_mid:.1e} col nmax-1={loss_top:.2e}"
            )
    print(
        "\n-- ENR space: expm of ENR generator vs analytic elements (2 modes, excitation cap Nexc)"
    )
    for Nexc in (6, 10, 16):
        for eta in (0.1, 0.3):
            dims = [Nexc + 1, Nexc + 1]
            a1, a2 = q.enr_destroy(dims, Nexc)
            Denr = (1j * eta * (a1 + a1.dag())).expm().full()
            nstates, state2idx, idx2state = q.enr_state_dictionaries(dims, Nexc)
            Dex_full = analytic_displacement(Nexc + 1, 1j * eta)
            Dref = np.zeros((nstates, nstates), dtype=complex)
            for (n1, n2), i in state2idx.items():
                for (m1, m2), j in state2idx.items():
                    if n2 == m2:
                        Dref[j, i] = Dex_full[m1, n1]
            diff = np.abs(Denr - Dref)
            # restrict comparison to states well inside the cap (n1+n2 <= Nexc/2)
            inner = [i for (n1, n2), i in state2idx.items() if n1 + n2 <= Nexc // 2]
            d_inner = np.max(diff[np.ix_(inner, inner)])
            print(
                f"Nexc={Nexc:2d} eta={eta:.1f} nstates={nstates:4d} | max|Denr-Dref| all={diff.max():.1e} inner(n1+n2<=Nexc/2)={d_inner:.1e} | "
                f"ENR expm unitarity defect={np.linalg.norm(Denr.conj().T @ Denr - np.eye(nstates), 2):.1e}"
            )


if __name__ == "__main__":
    bench_B2()
    bench_B1()
    bench_B1_qutip()
