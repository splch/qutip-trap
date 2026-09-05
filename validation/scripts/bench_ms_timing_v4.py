"""Wall time of a Moelmer-Soerensen pulse in the motion-Schroedinger picture at realistic joint dimensions
(2 ions, 1 to 3 modes), QuTiP 5.3.1, with the drive operator built as a TENSOR PRODUCT OF PER-MODE EXPONENTIALS.

v4 (2026-09-04, night). The 2026-09-04 numerics critique showed that v3's `gen.expm()` on the joint generator returns
a Dense Qobj, so v3 timed dense matrix-vector products and its nnz probe printed -1. v4 measures four constructions:
  construction 'csr'   : each D_m = expm[i eta_m (a_m + a_m^dag)] built in the single-mode space, converted to CSR and
                         tensored (PLAN.md Sections 5.1.1 and 11.3); nnz is checked against 2^(N-1) prod_m d_m^2 per ion.
  construction 'dense' : the joint-generator expm of v3 (Dense).
  merged = True        : one module-level coefficient function shared by all four drive terms, which QobjEvo merges into
                         ONE operator sum_i (sigma_+^i D_i + h.c.) with coefficient Omega cos(mu t) (the physically right
                         build: the tones and ions share the coefficient).
  merged = False       : four distinct (but picklable, module-level) coefficient functions, one per term, as v3's four
                         lambdas were, so nothing merges and the RHS applies four operators.
It counts right-hand-side evaluations through the coefficient calls, prints the merged element count, compares the
final states of the integrators and constructions, and uses module-level coefficient functions so that the same build
pickles under map="parallel". Same fixture as v3: two 171Yb+ ions, omega_x = 3.0, omega_y = 2.9, omega_z = 1.0 MHz;
modes x-COM 3.000, x-rock 2.828, y-COM 2.900 MHz; eta = 0.080, +-0.0824, 0.047; closure eta Omega/eps = 1/2 (K = 1).

Run:  uv run --python 3.13 --with qutip --with numpy --with scipy python bench_ms_timing_v4.py
"""

import time

import numpy as np
import qutip as q

q.settings.core["auto_tidyup"] = False

OMEGA = [2 * np.pi * 3.0e6, 2 * np.pi * 2.8284e6, 2 * np.pi * 2.9e6]
ETA = [(0.080, 0.080), (0.0824, -0.0824), (0.047, 0.047)]  # per (mode)(ion)
EPS = 2 * np.pi * 10e3  # gate detuning from the sideband, 100 us loop
CALLS = [0]  # coefficient-call counter


def two_tone(t, Om, mu):
    """0.5 Om (e^{-i mu t} + e^{+i mu t}) = Om cos(mu t): the bichromatic coefficient, module-level so it pickles."""
    CALLS[0] += 1
    return Om * np.cos(mu * t)


def two_tone_a(t, Om, mu):
    return two_tone(t, Om, mu)


def two_tone_b(t, Om, mu):
    return two_tone(t, Om, mu)


def two_tone_c(t, Om, mu):
    return two_tone(t, Om, mu)


def two_tone_d(t, Om, mu):
    return two_tone(t, Om, mu)


DISTINCT = [two_tone_a, two_tone_b, two_tone_c, two_tone_d]


def build(nmodes, nmax, construction, merged):
    """H(t) = H0 + sum_i f(t) [sigma_+^i D_i + h.c.] with D_i by per-mode expm (CSR) or joint expm (Dense)."""
    ions = 2
    idm = [q.qeye(nmax)] * nmodes
    ident_ions = [q.qeye(2)] * ions
    a1 = q.destroy(nmax)
    H0 = 0
    for m in range(nmodes):
        ops = list(idm)
        ops[m] = a1.dag() * a1
        H0 = H0 + OMEGA[m] * q.tensor(*ident_ions, *ops)
    H0 = H0.to("CSR")
    terms = [H0]
    mu = OMEGA[0] - EPS
    Om = EPS / (2 * ETA[0][0])  # eta Omega/eps = 1/2
    k = 0
    for i in range(ions):
        sp_ops = list(ident_ions)
        sp_ops[i] = q.sigmap()
        if construction == "csr":
            Dm = [
                (1j * ETA[m][i] * (a1 + a1.dag())).expm().to("CSR")
                for m in range(nmodes)
            ]
            V = q.tensor(*sp_ops, *Dm).to("CSR")
        else:  # v3 construction: exponential of the joint generator (returns Dense)
            a_list = []
            for m in range(nmodes):
                ops = list(idm)
                ops[m] = a1
                a_list.append(q.tensor(*ident_ions, *ops))
            gen = sum(
                1j * ETA[m][i] * (a_list[m] + a_list[m].dag()) for m in range(nmodes)
            )
            V = q.tensor(*sp_ops, *idm) * gen.expm()
        for op in (V, V.dag()):
            terms.append([op, two_tone if merged else DISTINCT[k]])
            k += 1
    H = q.QobjEvo(terms, args={"Om": Om, "mu": mu})
    Vdt = terms[1][0].dtype.__name__
    nnz_V = terms[1][0].to("CSR").data.as_scipy().nnz
    n_el = len(H.to_list())
    return H, Vdt, nnz_V, n_el


def run(nmodes, nmax, T, method, construction, merged):
    H, Vdt, nnz_V, n_el = build(nmodes, nmax, construction, merged)
    dim = H.shape[0]
    psi0 = q.tensor(q.basis(2, 1), q.basis(2, 1), *[q.basis(nmax, 0)] * nmodes)
    CALLS[0] = 0
    t0 = time.perf_counter()
    res = q.sesolve(
        H,
        psi0,
        [0, T],
        options={
            "method": method,
            "atol": 1e-10,
            "rtol": 1e-8,
            "nsteps": 10**7,
            "store_final_state": True,
        },
    )
    dt = time.perf_counter() - t0
    evals = (
        CALLS[0] // 4
    )  # every RHS evaluation calls the coefficient once per drive term (4 terms)
    return dim, dt, res.stats, Vdt, nnz_V, n_el, evals, res.final_state


if __name__ == "__main__":
    T = 20e-6
    print(
        "MS pulse, 2 ions, motion-Schroedinger picture, T = 20 us (scale x5 for 100 us), atol 1e-10 rtol 1e-8, closure eta Omega/eps = 1/2"
    )
    print(
        "nnz formula per ion: 2^(N-1) prod d_m^2; n_el = QobjEvo elements after merging equal coefficients; evals = RHS evaluations"
    )
    for nmodes, nmax in [(1, 12), (2, 8), (3, 6), (3, 8)]:
        finals = {}
        for construction in ("csr", "dense"):
            for merged in (True, False):
                for method in ("dop853", "vern9"):
                    try:
                        dim, dt, stats, Vdt, nnz_V, n_el, evals, psi = run(
                            nmodes, nmax, T, method, construction, merged
                        )
                        finals[(construction, merged, method)] = psi
                        pred = 2 * nmax ** (2 * nmodes)
                        per = 1e9 * dt / max(evals, 1) / (4 * nnz_V)
                        print(
                            f"modes={nmodes} nmax={nmax} dim={dim:5d} {construction:5s} merged={str(merged):5s} n_el={n_el} V.dtype={Vdt:5s} "
                            f"nnz(V)={nnz_V:8d} (formula {pred:8d}) {method:7s} wall={dt:7.2f}s evals={evals:6d} ns/(eval*4nnz)={per:6.2f}"
                        )
                    except Exception as e:  # report, do not hide
                        print(
                            f"modes={nmodes} nmax={nmax} {construction} merged={merged} {method}: FAILED {type(e).__name__}: {e}"
                        )
        ref = finals.get(("csr", True, "dop853"))
        if ref is not None:
            for key, psi in finals.items():
                if key == ("csr", True, "dop853"):
                    continue
                print(
                    f"    final-state check {key}: ||psi - psi_ref|| = {(psi - ref).norm():.2e}, |<ref|psi>|^2 = {abs(ref.overlap(psi)) ** 2:.12f}"
                )
    print("stats keys:", list(stats.keys()))
