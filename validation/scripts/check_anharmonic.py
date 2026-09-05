"""Does an off-resonant cubic (three-phonon) coupling of strength g = eps*omega accumulate a phase g*t (first order)
or g^2/Delta * t (second order)? Two modes a (2.0 MHz) and b, coupling g (a + a^dag)^2 (b + b^dag) with
g/2pi = 1.419 kHz (the plan's 40Ca+ value), resonance at omega_b = 2 omega_a; scan the mismatch Delta.

Run: uv run --with qutip python check_anharmonic.py
"""
import numpy as np
import qutip as qt

Na, Nb = 8, 6
a = qt.tensor(qt.destroy(Na), qt.qeye(Nb)); b = qt.tensor(qt.qeye(Na), qt.destroy(Nb))
g = 2 * np.pi * 1.419e3
wa = 2 * np.pi * 2.0e6
t = 100e-6
print(f"g t = {g*t:.3f} rad (the plan's first-order estimate for 100 us)")
for Delta_hz in (1e6, 0.3e6, 0.1e6, 0.03e6):
    wb = 2 * wa - 2 * np.pi * Delta_hz          # omega_b = 2 omega_a - Delta
    H0 = wa * a.dag() * a + wb * b.dag() * b
    H = H0 + g * (a + a.dag()) ** 2 * (b + b.dag())
    # phase of |1_a,0_b> relative to |0,0> beyond free evolution, from the propagator
    U = (-1j * H * t).expm(); U0 = (-1j * H0 * t).expm()
    ket10 = qt.tensor(qt.basis(Na, 1), qt.basis(Nb, 0)); ket00 = qt.tensor(qt.basis(Na, 0), qt.basis(Nb, 0))
    amp10 = (ket10.dag() * U * ket10) / (ket10.dag() * U0 * ket10)
    amp00 = (ket00.dag() * U * ket00) / (ket00.dag() * U0 * ket00)
    phase = np.angle(amp10 / amp00)
    leak = 1 - abs(ket10.dag() * U * ket10) ** 2
    # second-order estimate: dominant term couples |1,0> <-> |0,1>? (a^dag)^2 b etc. Use the dispersive form for the
    # near-resonant process a^2 b^dag: |1,0> is not coupled by a^2; the resonant pair is |2,0> <-> |0,1>.
    print(f"Delta/2pi = {Delta_hz/1e6:5.2f} MHz: phase(|1,0>) = {phase:+.5f} rad, population leaked {leak:.2e}; "
          f"g^2/(2pi Delta) t = {g**2/(2*np.pi*Delta_hz)*t:.4f} rad")
# also the state |2,0>, which IS coupled to |0,1> by a^2 b^dag (resonant at Delta = 0)
print("state |2,0> (coupled to |0,1> by a a b^dag):")
for Delta_hz in (1e6, 0.3e6, 0.1e6):
    wb = 2 * wa - 2 * np.pi * Delta_hz
    H0 = wa * a.dag() * a + wb * b.dag() * b
    H = H0 + g * (a + a.dag()) ** 2 * (b + b.dag())
    U = (-1j * H * t).expm(); U0 = (-1j * H0 * t).expm()
    ket20 = qt.tensor(qt.basis(Na, 2), qt.basis(Nb, 0)); ket00 = qt.tensor(qt.basis(Na, 0), qt.basis(Nb, 0))
    amp20 = (ket20.dag() * U * ket20) / (ket20.dag() * U0 * ket20)
    amp00 = (ket00.dag() * U * ket00) / (ket00.dag() * U0 * ket00)
    leak = 1 - abs(ket20.dag() * U * ket20) ** 2
    geff = g * np.sqrt(2)  # <0,1| g a a b^dag |2,0> = g sqrt(2)
    print(f"Delta/2pi = {Delta_hz/1e6:5.2f} MHz: phase = {np.angle(amp20/amp00):+.5f} rad, leaked {leak:.2e}; "
          f"geff^2/(2pi Delta) t = {geff**2/(2*np.pi*Delta_hz)*t:.4f} rad, (geff/2pi Delta)^2 = {(geff/(2*np.pi*Delta_hz))**2:.2e}")
