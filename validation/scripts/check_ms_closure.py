"""Which closure ratio eta*Omega/eps produces a maximally entangling MS gate when the exact bichromatic
Hamiltonian is written per tone in the plan's convention H_t = (hbar Omega/2) sigma_+ e^{i(eta(a+a^dag) - omega_t t)} + h.c.?
Two ions, one mode, tones at omega_eg -/+ delta with delta = nu - eps (both tones detuned eps from the sidebands),
one loop tau = 2 pi/eps. Reports concurrence and populations for eta*Omega/eps = 1/4 and 1/2, plus the
effective-Hamiltonian check H_eff = -(hbar eta Omega/2) S_y (a^dag e^{i eps t} + h.c.) derived from the same convention
and, as the NEGATIVE control of Section 9.16 row 4.4-4, the withdrawn doubled force -(hbar eta Omega) S_y (...), which
lands maximally entangling at eta*Omega/eps = 1/4 and on a product state at 1/2 - the factor-2 error the derivation
audit of 2026-09-04 caught."""
import numpy as np, qutip as qt

def run(ratio, eta=0.05, nu=2*np.pi*1.0e6, eps=2*np.pi*10e3, nmax=16, model="exact"):
    Omega = ratio*eps/eta
    a = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(nmax))
    sp = [qt.tensor(qt.sigmap(), qt.qeye(2), qt.qeye(nmax)), qt.tensor(qt.qeye(2), qt.sigmap(), qt.qeye(nmax))]
    sy = [qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(nmax)), qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(nmax))]
    D = (1j*eta*(a + a.dag())).expm()
    H0 = nu*a.dag()*a                      # qubit frame at omega_eg; motion in Schroedinger picture
    delta = nu - eps                        # tone detunings from the carrier: -/+ delta
    tau = 2*np.pi/eps
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(nmax, 0))  # |down,down,0>  (basis(2,1) = down for sigmap convention)
    if model == "exact":
        terms = [H0]
        for s in sp:
            A = 0.5*Omega*s*D              # (hbar Omega/2) sigma_+ D(i eta), coefficient e^{-i mu t} per tone, mu = -/+ delta
            terms.append([A, lambda t, args: np.exp(1j*delta*t) + np.exp(-1j*delta*t)])   # both tones share the operator
            terms.append([A.dag(), lambda t, args: np.exp(-1j*delta*t) + np.exp(1j*delta*t)])
        H = qt.QobjEvo(terms)
    else:
        Sy = sy[0] + sy[1]
        factor = 0.5 if model == "effective" else 1.0    # "doubled": the withdrawn displayed H, no 1/2
        Fop = -(eta*Omega*factor)*Sy*a.dag()
        H = qt.QobjEvo([[Fop, lambda t, args: np.exp(1j*eps*t)], [Fop.dag(), lambda t, args: np.exp(-1j*eps*t)]])
    res = qt.sesolve(H, psi0, [0, tau], options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7, "method": "dop853"})
    rho_spin = res.states[-1].ptrace([0, 1])
    pops = np.real(np.diag(rho_spin.full()))
    return Omega/(2*np.pi), qt.concurrence(rho_spin), pops

for label, model in (
    ("exact per-tone H, (hbar Omega/2) sigma_+ per tone", "exact"),
    ("effective H = -(hbar eta Omega/2) S_y (a^dag e^{i eps t} + h.c.)", "effective"),
    ("NEGATIVE CONTROL, withdrawn doubled H = -(hbar eta Omega) S_y (a^dag e^{i eps t} + h.c.)", "doubled"),
):
    print("==", label)
    for ratio in (0.25, 0.5):
        Om, C, pops = run(ratio, model=model)
        print(f"  eta*Omega/eps = {ratio}: Omega/2pi = {Om/1e3:.1f} kHz, concurrence = {C:.4f}, populations = {np.round(pops, 4)}")
