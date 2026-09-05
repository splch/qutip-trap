"""Wall-time and step count of a Moelmer-Soerensen pulse in the motion-Schroedinger
picture at realistic joint dimensions (2 ions, 1 to 3 modes), QuTiP 5.3.1.
"""

import time
import numpy as np
import qutip as q

q.settings.core["auto_tidyup"] = False


def ms_hamiltonian(nmodes, nmax, omega_modes, eta, Om, eps):
    """Two ions, bichromatic drive on both, tones at +-(omega_1 - eps) from the carrier."""
    ions = 2
    mode_ops = []
    idm = [q.qeye(nmax)] * nmodes
    for m in range(nmodes):
        ops = list(idm)
        ops[m] = q.destroy(nmax)
        mode_ops.append(ops)
    ident_ions = [q.qeye(2)] * ions
    H0 = 0
    D = 0
    a_list = []
    for m in range(nmodes):
        a = q.tensor(*ident_ions, *mode_ops[m])
        a_list.append(a)
        H0 = H0 + omega_modes[m] * a.dag() * a
    # displacement operator per ion (same eta for both ions, COM-like), exact expm
    gen = sum(1j * eta[m] * (a_list[m] + a_list[m].dag()) for m in range(nmodes))
    Dop = gen.expm()
    terms = [H0]
    mu = omega_modes[0] - eps  # beat-note detuning from carrier
    for i in range(ions):
        sp_ops = list(ident_ions)
        sp_ops[i] = q.sigmap()
        sig_p = q.tensor(*sp_ops, *idm)
        V = sig_p * Dop
        # two tones: e^{-i mu t} and e^{+i mu t}
        terms.append(
            [
                V,
                lambda t, Om=Om, mu=mu: (
                    0.5 * Om * (np.exp(-1j * mu * t) + np.exp(1j * mu * t))
                ),
            ]
        )
        terms.append(
            [
                V.dag(),
                lambda t, Om=Om, mu=mu: (
                    0.5 * Om * (np.exp(1j * mu * t) + np.exp(-1j * mu * t))
                ),
            ]
        )
    return q.QobjEvo(terms), a_list


def run(nmodes, nmax, T, method):
    omega = [2 * np.pi * 3.0e6, 2 * np.pi * 2.9e6, 2 * np.pi * 2.8e6][:nmodes]
    eta = [0.08, 0.06, 0.05][:nmodes]
    eps = 2 * np.pi * 10e3  # gate detuning from the sideband, 100 us loop
    Om = eps / (2 * eta[0])  # single-loop MS closure: eta Omega/eps = 1/2 (K=1)
    H, a_list = ms_hamiltonian(nmodes, nmax, omega, eta, Om, eps)
    dim = H.shape[0]
    psi0 = q.tensor(q.basis(2, 1), q.basis(2, 1), *[q.basis(nmax, 0)] * nmodes)
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
    nnz = H(0.0).data.as_scipy().nnz if hasattr(H(0.0).data, "as_scipy") else -1
    return dim, dt, res.stats, nnz


if __name__ == "__main__":
    T = 20e-6  # one fifth of a 100 us gate
    print(
        "MS pulse, 2 ions, motion-Schroedinger picture, T = 20 us (scale x5 for 100 us), atol 1e-10 rtol 1e-8"
    )
    for nmodes, nmax in [(1, 12), (2, 8), (3, 6), (3, 8)]:
        for method in ("vern9", "dop853"):
            try:
                dim, dt, stats, nnz = run(nmodes, nmax, T, method)
                keys = {
                    k: stats[k]
                    for k in stats
                    if k in ("run time", "num_steps", "method")
                }
                print(
                    f"modes={nmodes} nmax={nmax} dim={dim:5d} nnz(H0+V)={nnz} {method:7s} wall={dt:7.2f}s  stats={keys}"
                )
            except Exception as e:  # report, do not hide
                print(
                    f"modes={nmodes} nmax={nmax} {method}: FAILED {type(e).__name__}: {e}"
                )
    print("stats keys:", list(stats.keys()))
