"""Micromotion factor on the Lamb-Dicke parameter: exact Floquet function u(t) of the Mathieu oscillator
 x'' + (omega_rf^2/4)[a + 2 q cos(omega_rf t)] x = 0, in the RMP form x(t) = x0 [a u*(t) + a^dag u(t)].
Two normalizations are compared: (A) u(0) = 1 (RMP lowest-order form gives C0 = 1/(1+q/2));
(B) the canonical Wronskian normalization Im(u* u') = nu, which makes [x, p] = i hbar with x0 = sqrt(hbar/2 m nu).
The sideband coupling strength relative to eta = k x0 is |c0|, the e^{i nu t} Fourier coefficient of u(t)
under normalization (B). Reports (1+q/2)^-1, 1 + 3q^2/16 and the exact |c0| for a = 0 and q = 0.1, 0.2, 0.3."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def floquet(a, q, nper=64, N=4096):
    # dimensionless time xi = omega_rf t / 2: x'' + (a - 2 q cos 2 xi) x = 0; period pi
    def rhs(xi, y):
        return [y[1], -(a - 2*q*np.cos(2*xi))*y[0]]
    T = np.pi
    sol1 = solve_ivp(rhs, (0, T), [1, 0], rtol=1e-12, atol=1e-14, dense_output=True)
    sol2 = solve_ivp(rhs, (0, T), [0, 1], rtol=1e-12, atol=1e-14, dense_output=True)
    M = np.array([[sol1.y[0, -1], sol2.y[0, -1]], [sol1.y[1, -1], sol2.y[1, -1]]])
    beta = np.arccos(np.trace(M)/2)/np.pi          # cos(pi beta) = tr M / 2  (principal branch)
    # Floquet eigenvector: M v = e^{i pi beta} v
    w, V = np.linalg.eig(M)
    k = np.argmin(np.abs(w - np.exp(1j*np.pi*beta)))
    v = V[:, k]
    xi = np.linspace(0, T, N, endpoint=False)
    u = v[0]*sol1.sol(xi)[0] + v[1]*sol2.sol(xi)[0]
    du = v[0]*sol1.sol(xi)[1] + v[1]*sol2.sol(xi)[1]
    return beta, xi, u, du

for q in (0.1, 0.2, 0.3):
    a = 0.0
    beta, xi, u, du = floquet(a, q)
    nu = beta  # in units of omega_rf/2 ... with xi = omega_rf t/2, e^{i nu t} = e^{i beta xi}
    # normalization B: Im(u* du/dxi) = beta  (Wronskian 2 i beta in xi units)
    W = np.imag(np.conj(u)*du)         # constant in xi
    s = np.sqrt(beta/np.mean(W))
    uB = u*s
    # Fourier coefficient of e^{i beta xi} in uB: c0 = <uB e^{-i beta xi}>
    c0 = np.mean(uB*np.exp(-1j*beta*xi))
    # normalization A: u(0) = 1
    uA = u/u[0]
    c0A = np.mean(uA*np.exp(-1j*beta*xi))
    print(f"q = {q}: beta = {beta:.9f}, sqrt(a+q^2/2) = {np.sqrt(a+q*q/2):.9f}")
    print(f"   (1+q/2)^-1 = {1/(1+q/2):.6f};  1 + 3q^2/16 = {1+3*q*q/16:.6f}")
    print(f"   |c0| Wronskian-normalized = {abs(c0):.6f};  |c0| with u(0)=1 = {abs(c0A):.6f};  Im(uA* uA')/beta at 0 = {np.imag(np.conj(uA[0])*(du[0]/u[0]))/beta:.6f}")
