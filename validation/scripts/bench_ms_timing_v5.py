"""Wall time of the Section 11.1 Moelmer-Soerensen pulse with the drive operator held FACTORIZED (the matrix-free kernel of
Section 11.3 item 4, milestone M9b) against the assembled constructions of bench_ms_timing_v4.py.

v5 (2026-09-07). The same fixture as v3 and v4 (two 171Yb+ ions, omega_x = 3.0, omega_y = 2.9, omega_z = 1.0 MHz; modes x-COM
3.000, x-rock 2.828, y-COM 2.900 MHz; eta = 0.080, +-0.0824, 0.047; closure eta Omega/eps = 1/2, eps/2pi = 10 kHz; sesolve at
atol 1e-10, rtol 1e-8 from |dn dn>|0...0> for 20 us) with four constructions of the drive terms:
  factorized : sigma_+^i (x) prod_m D_m as a FactorizedOperator (qutip_trap.dynamics.kernels), one distinct coefficient per term
               as the real builder emits (QobjEvo.compress merges nothing, so the four terms stay four factorized applications);
  csr        : the tensor product of per-mode exponentials, CSR, distinct coefficients (v4 'csr merged=False');
  dense      : the same operator Dense, distinct coefficients (v4 'dense merged=False', v3's structure);
  merged     : one coefficient function shared by the four terms, which QobjEvo merges into ONE dense operator (v4 'dense
               merged=True', the best assembled construction of Section 11.1 and the column Sections 11.2 to 11.4 quote).
Part 1 times QobjEvo.matmul alone (one right-hand side, dispatch included) per construction over eleven spaces from dimension
48 to 8192, fits the two cost models of Section 11.2 (a factorized term ~ a + b per factor step + c per multiply-add; a CSR term
~ a' + b' per non-zero) and prints the constants qutip_trap.dynamics.kernels carries. Part 2 integrates the four Section 11.1
rows with dop853 and vern9, prints wall times, evaluation counts and the per-evaluation cost, and checks the final states of
every construction against the CSR one. Wall times are never compared by run_checks.py (bench_ scripts are timed only).

Run:  uv run python validation/scripts/bench_ms_timing_v5.py      (about four minutes on the plan's machine)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qutip_trap.dynamics.kernels import factorized_qobj, kernel_costs_us  # noqa: E402

qt.settings.core["auto_tidyup"] = False

OMEGA = [2 * np.pi * 3.0e6, 2 * np.pi * 2.8284e6, 2 * np.pi * 2.9e6]
ETA = [(0.080, 0.080), (0.0824, -0.0824), (0.047, 0.047)]  # per (mode)(ion)
EPS = 2 * np.pi * 10e3
CALLS = [0]


def two_tone(t, Om, mu, tag=None):
    """0.5 Om (e^{-i mu t} + e^{+i mu t}) = Om cos(mu t): the bichromatic coefficient, module-level so it pickles."""
    CALLS[0] += 1
    return Om * np.cos(mu * t)


def build(nmodes, nmax, construction):
    ions = 2
    dims = [2] * ions + [nmax] * nmodes
    a1 = qt.destroy(nmax)
    idm = [qt.qeye(nmax)] * nmodes
    ident_ions = [qt.qeye(2)] * ions
    H0 = 0
    for m in range(nmodes):
        ops = list(idm)
        ops[m] = a1.dag() * a1
        H0 = H0 + OMEGA[m] * qt.tensor(*ident_ions, *ops)
    terms = [H0.to("CSR")]
    mu = OMEGA[0] - EPS
    Om = EPS / (2 * ETA[0][0])
    for i in range(ions):
        Dm = [(1j * ETA[m][i] * (a1 + a1.dag())).expm() for m in range(nmodes)]
        if construction == "factorized":
            V = factorized_qobj(
                dims, {i: qt.sigmap().full(), **{ions + m: Dm[m].full() for m in range(nmodes)}}
            )
        else:
            sp_ops = list(ident_ions)
            sp_ops[i] = qt.sigmap()
            V = qt.tensor(*sp_ops, *[d.to("CSR") for d in Dm]).to("CSR")
            if construction in ("dense", "merged"):
                V = V.to("Dense")
        for j, op in enumerate((V, V.dag())):
            if construction == "merged":
                terms.append([op, qt.coefficient(two_tone, args={"Om": Om, "mu": mu})])
            else:
                terms.append([op, qt.coefficient(two_tone, args={"Om": Om, "mu": mu, "tag": (i, j)})])
    H = qt.QobjEvo(terms)
    n_el = len(H.to_list())
    return H, dims, n_el


def per_rhs_us(H, psi, n=300):
    H.matmul(0.0, psi)
    t0 = time.perf_counter()
    for k in range(n):
        H.matmul(1e-7 * k, psi)
    return 1e6 * (time.perf_counter() - t0) / n


def part1():
    print("== Part 1: one right-hand side (QobjEvo.matmul, dispatch included) per construction, microseconds")
    rows = []
    spaces = [
        ((2, 2, 12), 1, 12),
        ((2, 2, 8, 8), 2, 8),
        ((2, 2, 10, 11), 2, None),
        ((2, 2, 11, 13), 2, None),
        ((2, 2, 12, 12), 2, 12),
        ((2, 2, 6, 6, 6), 3, 6),
        ((2, 2, 8, 8, 8), 3, 8),
        ((2, 2, 2, 12, 12), 2, 12),
        ((2, 2, 2, 8, 8, 8), 3, 8),
        ((2, 2, 2, 2, 12, 12), 2, 12),
        ((2, 2, 2, 2, 8, 8, 8), 3, 8),
    ]
    for dims, nmodes, nmax in spaces:
        n_ions = len(dims) - nmodes
        D = int(np.prod(dims))
        psi = qt.rand_ket(list(dims), seed=1)
        # a general builder for unequal mode dimensions (the Bell fixture's [2, 2, 10, 11] and [2, 2, 11, 13])
        mode_dims = list(dims[n_ions:])
        mats = [(1j * 0.08 * (qt.destroy(d) + qt.destroy(d).dag())).expm() for d in mode_dims]
        h0 = (0.0 * qt.qeye(list(dims))).to("CSR")
        terms_f, terms_c = [h0], [h0]
        for i in range(n_ions):
            fac = {i: qt.sigmap().full(), **{n_ions + m: mats[m].full() for m in range(nmodes)}}
            V_f = factorized_qobj(dims, fac)
            ops = [qt.qeye(2)] * n_ions
            ops[i] = qt.sigmap()
            V_c = qt.tensor(*ops, *[m.to("CSR") for m in mats]).to("CSR")
            for j, (of, oc) in enumerate(((V_f, V_c), (V_f.dag(), V_c.dag()))):
                terms_f.append([of, qt.coefficient(two_tone, args={"Om": 1.0, "mu": 1e6, "tag": (i, j)})])
                terms_c.append([oc, qt.coefficient(two_tone, args={"Om": 1.0, "mu": 1e6, "tag": (i, j)})])
        hf, hc = qt.QobjEvo(terms_f), qt.QobjEvo(terms_c)
        tf, tc = per_rhs_us(hf, psi), per_rhs_us(hc, psi)
        err = (hf.matmul(2e-7, psi) - hc.matmul(2e-7, psi)).norm()
        n_terms = 2 * n_ions
        block = D // 2
        nnz_term = block * int(np.prod(mode_dims))
        macs_term = sum(block * d for d in mode_dims)
        a_model, f_model = kernel_costs_us(dims, 0, list(range(n_ions, len(dims))))
        rows.append((dims, n_terms, tf, tc, nnz_term, macs_term, nmodes))
        print(
            f"dims={str(dims):26s} D={D:5d} terms={n_terms} factorized {tf:8.1f} us  csr {tc:8.1f} us  "
            f"per term: f {tf / n_terms:6.1f} c {tc / n_terms:7.1f}  model f {f_model:6.1f} c {a_model:7.1f}  "
            f"nnz/term {nnz_term:8d} macs/term {macs_term:7d}  |Hf - Hc| psi = {err:.1e}"
        )
    A_f = np.array([[r[6], r[5]] for r in rows])
    y_f = np.array([r[2] / r[1] for r in rows])
    cf, *_ = np.linalg.lstsq(A_f, y_f, rcond=None)
    A_c = np.array([[r[4]] for r in rows])
    y_c = np.array([r[3] / r[1] for r in rows])
    cc, *_ = np.linalg.lstsq(A_c, y_c, rcond=None)
    print(
        f"fit (no intercept): factorized per term ~ {cf[0]:.2f} us per factor step + {1e3 * cf[1]:.3f} ns per multiply-add"
    )
    print(f"fit (no intercept): csr per term ~ {1e3 * cc[0]:.3f} ns per non-zero")
    print(
        "constants carried by qutip_trap.dynamics.kernels: term 2.0 us, step 2.5 us, 0.6 ns/MAC; csr 0.5 us + 0.57 ns/nnz"
    )


def run(nmodes, nmax, T, method, construction):
    H, dims, n_el = build(nmodes, nmax, construction)
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), *[qt.basis(nmax, 0)] * nmodes)
    CALLS[0] = 0
    t0 = time.perf_counter()
    res = qt.sesolve(
        H,
        psi0,
        [0, T],
        options={"method": method, "atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7, "store_final_state": True},
    )
    dt = time.perf_counter() - t0
    per_rhs = 1 if construction == "merged" else 4
    return dims, dt, CALLS[0] // per_rhs, n_el, res.final_state


def part2():
    T = 20e-6
    print(
        "\n== Part 2: the Section 11.1 rows, T = 20 us (scale x5 for 100 us), atol 1e-10 rtol 1e-8, closure eta Omega/eps = 1/2"
    )
    for nmodes, nmax in [(1, 12), (2, 8), (3, 6), (3, 8)]:
        finals = {}
        for construction in ("factorized", "csr", "dense", "merged"):
            for method in ("dop853", "vern9"):
                if construction == "dense" and nmax == 8 and nmodes == 3 and method == "vern9":
                    continue  # 130 s in v4; the dop853 row suffices
                dims, dt, evals, n_el, psi = run(nmodes, nmax, T, method, construction)
                finals[(construction, method)] = psi
                D = int(np.prod(dims))
                print(
                    f"modes={nmodes} nmax={nmax} dim={D:5d} {construction:10s} n_el={n_el} {method:7s} wall={dt:7.2f}s "
                    f"evals={evals:6d} us/eval={1e6 * dt / max(evals, 1):8.1f}"
                )
        ref = finals[("csr", "dop853")]
        for key, psi in finals.items():
            if key == ("csr", "dop853"):
                continue
            print(f"    final-state check {key}: ||psi - psi_ref|| = {(psi - ref).norm():.2e}")


if __name__ == "__main__":
    part1()
    part2()
