"""Independent check of the thesis Eq. (5.19) blackbody deshelving rate for 88Sr+ D5/2."""
import numpy as np

h, hbar, kB, c = 6.62607015e-34, 1.054571817e-34, 1.380649e-23, 299792458.0
# Section 13 / PLAN.md 4.5.7: lambda is the VACUUM wavelength. NIST JPCRD Table 2 prints the AIR value
# 10 327.309 A; the vacuum wavelength from the NIST ASD levels 4d 2D5/2 = 14 836.24 cm^-1 and
# 5p 2P3/2 = 24 516.65 cm^-1 is 1e7/9680.41 = 1033.0141 nm (276 ppm larger, so hbar*omega/kT shifts by
# 276 ppm and n_bar by 1.4%; immaterial to the eleven-orders-of-magnitude conclusion, but the convention).
lam_PD = 1033.0141e-9      # m, D5/2 - P3/2 VACUUM (air 1032.7309 nm, NIST JPCRD Table 2)
A_PD   = 8.7e6             # s^-1, NIST JPCRD Table 2 ref 67GAL; same value used in thesis Sec 5.6.2/5.6.3
JP, JD = 1.5, 2.5
b      = 0.06              # branch from 5p P3/2 back to 4d D5/2 (so (1-b) leaves the shelf)

omega = 2*np.pi*c/lam_PD
deg   = (2*JP+1)/(2*JD+1)          # 4/6
print("D5/2 - P3/2 : lambda = %.4f nm, hbar*omega = %.4f eV" % (lam_PD*1e9, hbar*omega/1.602176634e-19))
print("degeneracy factor (2Jp+1)/(2Jd+1) = %.4f" % deg)
print()
print(f"{'T /K':>7} {'hbar w/kT':>10} {'n_bar(BBR)':>12} {'rate /s^-1':>13} {'vs A=2.559 s^-1':>18}")
for T in (300.0, 293.0, 400.0, 1000.0, 2000.0):
    x = hbar*omega/(kB*T)
    nbar = 1.0/(np.expm1(x))
    rate = (1-b)*deg*A_PD*nbar
    print(f"{T:7.0f} {x:10.3f} {nbar:12.3e} {rate:13.3e} {rate/2.558854:18.2e}")
print()
T = 300.0
x = hbar*omega/(kB*T); nbar = 1/np.expm1(x)
rate = (1-b)*deg*A_PD*nbar
print("At T = 300 K:  rate = %.2e s^-1  (thesis Sec. 5.6.3 states 'a quenching rate of 10^-14 s^-1')" % rate)
print("  fractional effect on tau = %.1e   -> shifts 390.8 ms by %.2e ms" % (rate/2.558854, 390.8*rate/2.558854))
print("  i.e. %d orders of magnitude below the 1.6 ms measurement uncertainty" % round(np.log10(1.6/(390.8*rate/2.558854))))
