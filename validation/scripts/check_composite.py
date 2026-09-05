"""Composite pulses, dynamical decoupling and the filter-function description of control noise.

Backs the [recomputed here] numbers of PLAN.md Sections 4.3.5 and 6.9 and the Section 9.15 table.
Run:  uv run --python 3.13 --with qutip --with numpy --with scipy --with sympy --with mpmath \
          python check_composite.py

Conventions (PLAN.md Section 13): R(theta, phi) = exp[-i theta rho(phi).sigma/2] with
rho(phi) = (cos phi, sin phi, 0), so a 2*pi segment is exactly -I and a carrier pi pulse is Omega t = pi.
The static error primitive is M(theta, phi; eps_a, eps_d) = exp[-i theta{(1+eps_a) rho(phi).sigma
+ eps_d sigma_z}/2].  Sequences are listed in TIME order and folded right-to-left (first listed acts
first).  Fidelities: F_C = |Tr(U_id^dag U)|/2, F_K = F_C^2, F_avg = (2 F_C^2 + 1)/3.
Spectra are two-sided in angular frequency with the exp(-i omega t) kernel; the filter-function
weight is 1/omega^2 paired with an explicit -i omega inside the frequency-domain control matrix.
Everything runs in mpmath so that the high-order cancellations are not roundoff-limited.
"""

import mpmath as mp
from itertools import combinations

mp.mp.dps = 60
PI = mp.pi
I2 = mp.matrix([[1, 0], [0, 1]])
SX = mp.matrix([[0, 1], [1, 0]])
SY = mp.matrix([[0, -1j], [1j, 0]])
SZ = mp.matrix([[1, 0], [0, -1]])


# --------------------------------------------------------------------------------------------
# 0. primitive, folding, metrics
# --------------------------------------------------------------------------------------------
def M(theta, phi, ea=0, ed=0):
    """M(theta, phi; eps_a, eps_d) = exp[-i theta{(1+ea) rho(phi).sigma + ed sigma_z}/2] (exact)."""
    nx = (1 + ea) * mp.cos(phi)
    ny = (1 + ea) * mp.sin(phi)
    nz = mp.mpf(ed)
    n = mp.sqrt(nx * nx + ny * ny + nz * nz)
    if n == 0:
        return I2.copy()
    A = (nx * SX + ny * SY + nz * SZ) / n
    return mp.cos(theta * n / 2) * I2 - 1j * mp.sin(theta * n / 2) * A


def R(theta, phi):
    return M(theta, phi)


def fold(seq, ea=0, ed=0):
    """seq = TIME-ORDERED [(theta, phi), ...]; U = M_last ... M_first."""
    U = I2.copy()
    for th, ph in seq:
        U = M(th, ph, ea, ed) * U
    return U


def fold_addr(seq, eN):
    """Addressing model: an unaddressed spin sees R(theta*eps_N, phi); the target is the identity."""
    U = I2.copy()
    for th, ph in seq:
        U = M(th * eN, ph) * U
    return U


def trace_overlap(U, V):
    return (
        U[0, 0].conjugate() * V[0, 0]
        + U[0, 1].conjugate() * V[0, 1]
        + U[1, 0].conjugate() * V[1, 0]
        + U[1, 1].conjugate() * V[1, 1]
    )


def F_C(U, V):
    return abs(trace_overlap(U, V)) / 2


def F_K(U, V):
    return F_C(U, V) ** 2


def F_avg(U, V):
    return (2 * F_C(U, V) ** 2 + 1) / 3


def specnorm(A):
    B = A.H * A
    tr = B[0, 0] + B[1, 1]
    det = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]
    return mp.sqrt(abs((tr + mp.sqrt(tr * tr - 4 * det)) / 2))


def slope(f, e1, e2):
    return mp.log(f(e1) / f(e2)) / mp.log(e1 / e2)


def head(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100)


# --------------------------------------------------------------------------------------------
# 1. the composite-pulse library, every sequence a product of R(theta, phi) in TIME order
# --------------------------------------------------------------------------------------------
def arcsinc(y):
    """inverse of sin(x)/x on the branch (0, pi], arcsinc(0) = pi."""
    if y == 0:
        return PI
    return mp.findroot(lambda x: mp.sin(x) / x - y, mp.mpf("1.5"))


def seq_primitive(th, phit=0):
    return [(th, phit)]


def seq_sk1(th, phit=0):
    """SK1 corrected: U_1X(-theta) M_0(theta), phi_1 = arccos(-theta/(4 pi))."""
    p = mp.acos(-th / (4 * PI))
    return [(th, phit), (2 * PI, phit + p), (2 * PI, phit - p)]


def seq_sk1_printed(th, phit=0):
    """BHC Eq. (10) AS PRINTED: U_1X(+theta) M_0(theta) -- first order, worse than a bare pulse."""
    p = mp.acos(th / (4 * PI))
    return [(th, phit), (2 * PI, phit - p), (2 * PI, phit + p)]


def seq_bb1(th, phit=0):
    """BB1 = B2 = W1: phi_B2 = arccos(-theta/(4 pi)), middle pulse at THREE times that phase."""
    p = mp.acos(-th / (4 * PI))
    return [(th, phit), (PI, phit + p), (2 * PI, phit + 3 * p), (PI, phit + p)]


def seq_nb1(th, phit=0):
    """NB1 = N2: identical to B2 except the middle phase, -phi instead of 3 phi."""
    p = mp.acos(-th / (4 * PI))
    return [(th, phit), (PI, phit + p), (2 * PI, phit - p), (PI, phit + p)]


def seq_pb1(th, phit=0):
    """PB1 = P2 = PD2: FIVE pulses (target plus a four-pulse symmetric corrector)."""
    p = mp.acos(-th / (8 * PI))
    return [
        (th, phit),
        (2 * PI, phit + p),
        (2 * PI, phit - p),
        (2 * PI, phit - p),
        (2 * PI, phit + p),
    ]


def seq_pb1_3corr(th, phit=0):
    """NEGATIVE CONTROL: P2's phase kept but forced into a THREE-pulse corrector."""
    p = mp.acos(-th / (8 * PI))
    return [(th, phit), (2 * PI, phit + p), (2 * PI, phit - p), (2 * PI, phit + p)]


def corpse_angles(th, n=(1, 1, 0)):
    k = mp.asin(mp.sin(th / 2) / 2)
    return (
        2 * n[0] * PI + th / 2 - k,
        2 * n[1] * PI - 2 * k,
        2 * n[2] * PI + th / 2 - k,
    )


def seq_corpse(th, phit=0, n=(1, 1, 0)):
    t1, t2, t3 = corpse_angles(th, n)
    return [(t1, phit), (t2, phit + PI), (t3, phit)]


def seq_scorpse(th, phit=0):
    return seq_corpse(th, phit, n=(0, 1, 0))


def seq_scrofulous(th, phit=0):
    t1 = arcsinc(2 * mp.cos(th / 2) / PI)
    p1 = mp.acos(-PI * mp.cos(t1) / (2 * t1 * mp.sin(th / 2)))
    p2 = p1 - mp.acos(-PI / (2 * t1))
    return [(t1, phit + p1), (PI, phit + p2), (t1, phit + p1)]


def seq_cinsk(th, phit=0):
    t1, t2, t3 = corpse_angles(th)
    p = mp.acos(-th / (4 * PI))
    return [
        (t1, phit),
        (t2, phit + PI),
        (t3, phit),
        (2 * PI, phit - p),
        (2 * PI, phit + p),
    ]


def seq_cinbb(th, phit=0):
    t1, t2, t3 = corpse_angles(th)
    p = mp.acos(-th / (4 * PI))
    return [
        (t1, phit),
        (t2, phit + PI),
        (t3, phit),
        (PI, phit + p),
        (2 * PI, phit + 3 * p),
        (PI, phit + p),
    ]


# --- palindromic Trotter-Suzuki ladder (Merrill-Brown Eqs. 44-50) ---------------------------
def _seg(area, phase):
    """emit a signed area as a POSITIVE area with a phase shift of pi where needed."""
    return (abs(area), phase if area >= 0 else phase + PI)


def T1(k, phi):
    a = 2 * k * PI
    return [_seg(a, -phi), _seg(a, phi)][
        ::-1
    ]  # M(2k pi,-phi) M(2k pi,phi), time order reversed


def T2(k, phi, family="P"):
    if family == "P":
        return T1(k, phi) + T1(k, -phi)
    if family == "N":
        return T1(k / 2, -phi) + T1(k / 2, phi)
    if family == "B":
        if abs(k) % 2 == 0:
            return T1(k / 2, phi) + T1(k / 2, -phi)
        a = k * PI  # odd k, CORRECTED four-pulse palindromic layer
        return [_seg(a, phi), _seg(a, 3 * phi), _seg(a, 3 * phi), _seg(a, phi)]
    if family == "B_printed":
        if abs(k) % 2 == 0:
            return T1(k / 2, phi) + T1(k / 2, -phi)
        a = k * PI  # odd k AS PRINTED: only three pulses
        return [_seg(a, phi), _seg(a, 3 * phi), _seg(a, 3 * phi)]
    raise ValueError(family)


def T2j(j, k, phi, family="P"):
    if j == 1:
        return T2(k, phi, family)
    inner = T2j(j - 1, k, phi, family)
    return (
        inner * 2 ** (2 * j - 2)
        + T2j(j - 1, -2 * k, phi, family)
        + inner * 2 ** (2 * j - 2)
    )


def f_ladder(j, f1, printed=False):
    f = mp.mpf(f1)
    for m in range(2, j + 1):
        f *= 2 ** (2 * m - 1) - (1 if printed else 2)
    return f


def seq_P2j(j, th, family="P", printed=False):
    f1 = 4 if family == "P" else 2
    fj = f_ladder(j, f1, printed)
    phi = mp.acos(-th / (2 * PI * fj))
    return [(th, mp.mpf(0))] + T2j(j, 1, phi, family), fj


head(
    "1. Order ladder: log-log slopes of 1 - F_K at theta = pi under the static primitive"
)
th = PI
V = M(th, 0)
e1, e2 = mp.mpf("1e-5"), mp.mpf("1e-6")
print(
    f"{'sequence':13s} {'pulses':>6s} {'amp slope':>10s} {'det slope':>10s} {'simul slope':>12s}"
    f"  {'total rotation/pi':>18s}"
)
LIB = [
    ("primitive", seq_primitive),
    ("SK1", seq_sk1),
    ("BB1 = B2", seq_bb1),
    ("PB1 = P2", seq_pb1),
    ("CORPSE", seq_corpse),
    ("short-CORPSE", seq_scorpse),
    ("SCROFULOUS", seq_scrofulous),
    ("red. CinSK", seq_cinsk),
    ("red. CinBB", seq_cinbb),
]
for name, sf in LIB:
    s = sf(th)
    fa = lambda e: 1 - F_K(V, fold(s, ea=e))
    fd = lambda e: 1 - F_K(V, fold(s, ed=e))
    fs = lambda e: 1 - F_K(V, fold(s, ea=e, ed=e))
    tot = sum(x[0] for x in s) / PI
    print(
        f"{name:13s} {len(s):6d} {mp.nstr(slope(fa, e1, e2), 5):>10s} {mp.nstr(slope(fd, e1, e2), 5):>10s} "
        f"{mp.nstr(slope(fs, e1, e2), 5):>12s}  {mp.nstr(tot, 6):>18s}"
    )
print(
    "expected: primitive 2/2, SK1 4/2, BB1 6/2, CORPSE 2/4, SCROFULOUS 4/2, CinSK & CinBB 4 simultaneous"
)

head("2. Leading coefficients at theta = pi (eps = 1e-12, dps raised per order)")
print(
    "   a slope-6 sequence sits at 1-F_K ~ 1e-72 there and underflows the default 60 digits"
)
_dps0 = mp.mp.dps
e = mp.mpf("1e-12")
rows = [
    ("primitive", seq_primitive, "a", 2),
    ("primitive", seq_primitive, "d", 2),
    ("SK1", seq_sk1, "a", 4),
    ("BB1 = B2", seq_bb1, "a", 6),
    ("PB1 = P2", seq_pb1, "a", 6),
    ("SCROFULOUS", seq_scrofulous, "a", 4),
    ("SCROFULOUS", seq_scrofulous, "d", 2),
    ("CORPSE", seq_corpse, "d", 4),
    ("red. CinSK", seq_cinsk, "d", 4),
    ("red. CinBB", seq_cinbb, "d", 4),
    ("red. CinBB", seq_cinbb, "a", 6),
    ("red. CinSK", seq_cinsk, "a", 4),
]
for name, sf, chan, p in rows:
    mp.mp.dps = 12 * p + 60
    e = mp.mpf("1e-12")
    Vp = M(mp.pi, 0)
    kw = {"ea": e} if chan == "a" else {"ed": e}
    U = fold(sf(mp.pi), **kw)
    print(
        f"  {name:12s} {chan}: (1-F_K)/eps^{p} = {mp.nstr(1 - F_K(Vp, U), 12):>18s} / eps^{p} -> "
        f"{mp.nstr((1 - F_K(Vp, U)) / e**p, 12):>18s}   (1-F_C form: {mp.nstr((1 - F_C(Vp, U)) / e**p, 12)})"
    )
mp.mp.dps = _dps0
PI = mp.pi
th = PI
V = M(th, 0)
print(
    f"  closed forms: SK1 (pi^2 sin 2 phi_1)^2 = "
    f"{mp.nstr((PI**2 * mp.sin(2 * mp.acos(-mp.mpf(1) / 4))) ** 2, 12)}  with sin 2 phi_1 = -sqrt(15)/8 = "
    f"{mp.nstr(-mp.sqrt(15) / 8, 10)}"
)
print(
    f"  primitive amplitude infidelity is EXACT: 1-F_K = sin^2(theta eps_a/2); at eps=1e-3, "
    f"{mp.nstr(1 - F_K(V, fold([(th, 0)], ea=mp.mpf('1e-3'))), 12)} vs "
    f"{mp.nstr(mp.sin(th * mp.mpf('1e-3') / 2) ** 2, 12)}"
)
print(
    f"  primitive detuning:   1-F_K = (1/2)(1-cos theta) eps_d^2 + O(eps^4) -> 1 at theta = pi"
)

head(
    "3. Fidelity-measure relations, and the two printed forms that are identically constant"
)
U = fold(seq_sk1(th), ea=mp.mpf("0.03"))
fc, fk, fav = F_C(V, U), F_K(V, U), F_avg(V, U)
print(
    f"  F_C = {mp.nstr(fc, 15)}   F_C^2 = {mp.nstr(fc**2, 15)}   F_K = {mp.nstr(fk, 15)}   "
    f"|F_K - F_C^2| = {mp.nstr(abs(fk - fc**2), 4)}"
)
print(
    f"  F_avg = {mp.nstr(fav, 15)}   (2 F_C^2 + 1)/3 = {mp.nstr((2 * fc**2 + 1) / 3, 15)}"
)
print(
    f"  1 - F_K = {mp.nstr(1 - fk, 12)} against 2(1 - F_C) = {mp.nstr(2 * (1 - fc), 12)}  "
    f"(ratio {mp.nstr((1 - fk) / (2 * (1 - fc)), 10)}, -> 1 as eps -> 0)"
)
print(
    f"  1 - F_avg = {mp.nstr(1 - fav, 12)} against (4/3)(1 - F_C) = {mp.nstr(4 * (1 - fc) / 3, 12)}"
)
print(
    f"  LYC's printed F = ||U V^dag||_2 for unitaries = {mp.nstr(specnorm(U * V.H), 15)}  (identically 1)"
)
print(
    f"  BHC's printed 1 - ||V^dag U||_2               = {mp.nstr(1 - specnorm(V.H * U), 4)}  (identically 0)"
)
print(
    "  order n  <=>  residual O(eps^{n+1})  <=>  distance slope n+1  <=>  infidelity slope 2(n+1)"
)

head(
    "4. NEGATIVE CONTROL: the printed SK1 sign (BHC Eq. 10), spectral-norm distance at theta = pi/2"
)
thh = PI / 2
Vh = M(thh, 0)
print(f"{'sequence':16s} {'eps=1e-2':>14s} {'1e-3':>14s} {'1e-4':>14s} {'slope':>8s}")
for name, sf in [
    ("bare pulse", seq_primitive),
    ("SK1 AS PRINTED", seq_sk1_printed),
    ("SK1 corrected", seq_sk1),
    ("BB1", seq_bb1),
    ("PB1", seq_pb1),
]:
    v = [specnorm(fold(sf(thh), ea=mp.mpf(x)) - Vh) for x in ("1e-2", "1e-3", "1e-4")]
    print(
        f"{name:16s} "
        + "".join(f"{mp.nstr(x, 6):>15s}" for x in v)
        + f" {mp.nstr(mp.log(v[0] / v[1]) / mp.log(mp.mpf(10)), 5):>8s}"
    )
print(
    f"  bare closed form 2|sin(theta eps/4)| at eps = 1e-2: "
    f"{mp.nstr(2 * abs(mp.sin(thh * mp.mpf('1e-2') / 4)), 8)}"
)
print(
    "  the printed U_1X(+theta) form is exactly TWICE the bare-pulse error: the correction adds."
)
print(
    f"  ceiling trap: ceil(-theta/(4 pi)) = {int(mp.ceil(-thh / (4 * PI)))} for 0 < theta <= 4 pi "
    f"(zero denominator); ceil(|a|/(4 pi)) = {int(mp.ceil(abs(-thh) / (4 * PI)))}"
)

head(
    "5. NEGATIVE CONTROL: the Trotter-Suzuki recursion factor f_j = (2^{2j-1} - 2) f_{j-1}"
)
print(
    "   (theta = pi/2; P2j and B2j scored against amplitude error, N2j against addressing error)"
)
for fam, model in (("P", "amp"), ("B", "amp"), ("N", "addr")):
    for printed in (False, True):
        s, fj = seq_P2j(2, thh, family=fam, printed=printed)
        if model == "amp":
            g = lambda x: 1 - F_K(Vh, fold(s, ea=x))
        else:
            g = lambda x: 1 - F_K(I2, fold_addr(s, x))
        v1, v2 = g(mp.mpf("0.05")), g(mp.mpf("0.025"))
        tag = "PRINTED (2^{2j-1}-1)" if printed else "corrected (2^{2j-1}-2)"
        print(
            f"  {fam}4 {tag:22s} f_2 = {int(fj):3d}  phi = arccos(-theta/{int(2 * fj)} pi)  "
            f"1-F_K(0.05) = {mp.nstr(v1, 6):>13s}  slope {mp.nstr(mp.log(v1 / v2) / mp.log(mp.mpf(2)), 5):>8s}"
            f"  ({len(s)} pulses)"
        )
print(
    f"  anchors the recursion must reproduce: phi_P2 = arccos(-theta/8 pi) [f_1 = 4], "
    f"phi_P4 = arccos(-theta/48 pi) [f_2 = 24]; f_j for P2j: "
    f"{[int(f_ladder(j, 4)) for j in (1, 2, 3, 4)]}, for N2j/B2j: {[int(f_ladder(j, 2)) for j in (1, 2, 3, 4)]}"
)

head(
    "6. NEGATIVE CONTROL: the odd-k B2j bottom layer (three printed pulses versus four)"
)
phi = mp.acos(-thh / (4 * PI))
eq43 = seq_bb1(thh)
corr = [(thh, 0)] + T2(1, phi, "B")
prin = [(thh, 0)] + T2(1, phi, "B_printed")
ea = mp.mpf("0.05")
print(
    f"  ||corrected 4-pulse layer - Eq.43 corrector|| at eps_A = 0.05: "
    f"{mp.nstr(specnorm(fold(corr[1:], ea=ea) - fold(eq43[1:], ea=ea)), 6)} "
    f"(the two middle 3 phi pulses merge into M(2 pi, 3 phi))"
)
print(f"{'variant':22s} {'1-F_K':>14s} {'1-F_C':>14s} {'slope':>8s}")
for nm, s in [
    ("Eq.43 B2", eq43),
    ("corrected odd-k", corr),
    ("PRINTED 3-pulse", prin),
    ("uncorrected pulse", [(thh, 0)]),
]:
    v1 = 1 - F_K(Vh, fold(s, ea=ea))
    v2 = 1 - F_K(Vh, fold(s, ea=ea / 2))
    print(
        f"{nm:22s} {mp.nstr(v1, 6):>14s} {mp.nstr(1 - F_C(Vh, fold(s, ea=ea)), 6):>14s} "
        f"{mp.nstr(mp.log(v1 / v2) / mp.log(mp.mpf(2)), 5):>8s}"
    )
print(
    "  the printed layer is not palindromic and is ~1200x WORSE than no correction at all."
)
print(
    f"  P2 structure discriminator at eps_A = 0.1: five-pulse P2 1-F_C = "
    f"{mp.nstr(1 - F_C(Vh, fold(seq_pb1(thh), ea=mp.mpf('0.1'))), 6)}, the same phase forced into a "
    f"THREE-pulse corrector {mp.nstr(1 - F_C(Vh, fold(seq_pb1_3corr(thh), ea=mp.mpf('0.1'))), 6)}, "
    f"bare pulse {mp.nstr(1 - F_C(Vh, fold([(thh, 0)], ea=mp.mpf('0.1'))), 6)} "
    f"(the mistake is ~3700x worse than P2 and ~16x worse than no correction)"
)

head(
    "7. BB1's phi = arccos(-theta/4 pi) reproduces the plan's B2 sequence and its general axis"
)
for tht in ("0.5", "1.0", "1.5"):
    tht = mp.mpf(tht) * PI
    p = mp.acos(-tht / (4 * PI))
    plan = [
        (tht, 0),
        (PI, p),
        (2 * PI, 3 * p),
        (PI, p),
    ]  # PLAN.md Section 4.3.5, read in time order
    mb43 = seq_bb1(tht)
    ea = mp.mpf("0.07")
    print(
        f"  theta_t = {mp.nstr(tht / PI, 4)} pi: phi_B2 = {mp.nstr(p, 10)} rad = "
        f"{mp.nstr(p * 180 / PI, 8)} deg; ||plan - MB Eq.43|| = "
        f"{mp.nstr(specnorm(fold(plan, ea=ea) - fold(mb43, ea=ea)), 4)}; "
        f"ideal-pulse residual ||U - R|| = {mp.nstr(specnorm(fold(mb43) - M(tht, 0)), 4)}"
    )
phit = mp.mpf("0.7")
Zt = mp.matrix(
    [[mp.e ** (-1j * phit / 2), 0], [0, mp.e ** (1j * phit / 2)]]
)  # Z(alpha) = exp(-i alpha Z/2)
ea = mp.mpf("0.05")
Ua = fold(seq_bb1(th, phit), ea=ea)
Ub = Zt * fold(seq_bb1(th, 0), ea=ea) * Zt.H
print(
    f"  general target axis: ||U_T^(phi_0) - Z(phi_0) U_T^(0) Z(phi_0)^dag|| = "
    f"{mp.nstr(specnorm(Ua - Ub), 4)} at phi_0 = 0.7 (ADD the offset to every phase, zeroth pulse included)"
)
print(
    f"  W1 = R(pi,phi)R(2 pi,3 phi)R(pi,phi) at zero error: ||W1 - I|| = "
    f"{mp.nstr(specnorm(fold(seq_bb1(th)[1:]) - I2), 4)}, ||W1 + I|| = "
    f"{mp.nstr(specnorm(fold(seq_bb1(th)[1:]) + I2), 4)} -- it is +I, NOT -I as the brief's check claims"
)
print(
    f"    (two pi pulses give (-i sigma)^2 = -I and the 2 pi filler gives -I, so the product is +I; "
    f"a lone 2 pi segment IS -I: ||M(2 pi, 0.7) + I|| = "
    f"{mp.nstr(specnorm(M(2 * PI, mp.mpf('0.7')) + I2), 4)})"
)

head("8. Cross-paper checks: Wimperis / Cummins / Merrill-Brown / Low-Yoder-Chuang")
print("  CORPSE angles against Cummins Table I (degrees, n = (1,1,0)):")
for d in (30, 45, 90, 180):
    t = mp.mpf(d) * PI / 180
    a = corpse_angles(t)
    print(
        f"    theta = {d:3d} deg -> ({mp.nstr(a[0] * 180 / PI, 7)}, {mp.nstr(a[1] * 180 / PI, 7)}, "
        f"{mp.nstr(a[2] * 180 / PI, 7)})"
    )
print(
    f"  SCROFULOUS at theta = 180 deg -> 180_{mp.nstr(seq_scrofulous(PI)[0][1] * 180 / PI, 4)} "
    f"180_{mp.nstr((seq_scrofulous(PI)[1][1] * 180 / PI) % 360, 6)} "
    f"180_{mp.nstr(seq_scrofulous(PI)[2][1] * 180 / PI, 4)}  (Tycko's 180_60 180_300 180_60)"
)
print("  N2 in the addressing model equals B2 in the amplitude model, numerically:")
for x in ("0.2", "0.1", "0.05"):
    x = mp.mpf(x)
    print(
        f"    eps = {mp.nstr(x, 3):>6s}: N2 addressing 1-F_C = "
        f"{mp.nstr(1 - F_C(I2, fold_addr(seq_nb1(thh), x)), 8):>14s}   B2 amplitude 1-F_C = "
        f"{mp.nstr(1 - F_C(Vh, fold(seq_bb1(thh), ea=x)), 8):>14s}"
    )
print(
    "  eps-free certificate Phi_L^j(phi) = f_L^j(gamma) for 0 < j <= n at theta_0 = 2 pi:"
)


def Phi(phis, j):
    tot = mp.mpc(0)
    for sub in combinations(range(len(phis)), j):
        acc = mp.mpf(0)
        for k, h in enumerate(sub, start=1):
            acc += (-1) ** k * phis[h]
        tot += mp.e ** (-1j * acc)
    return tot


def fLj(L, j, gam):
    T = (gam + L) / 2
    return mp.fsum(
        (-1) ** k * mp.binomial(T, k) * mp.binomial(L - T, j - k) for k in range(j + 1)
    )


for gam in ("0.25", "0.5", "1.0"):
    g = mp.mpf(gam)
    ap1 = [mp.acos(-g / 2), -mp.acos(-g / 2)]  # AP1 = SK1
    q = mp.acos(-g / 4)
    pd2 = [q, -q, -q, q]  # PD2 = PB1 (palindromic)
    for nm, ph, n in (("AP1", ap1, 1), ("PD2", pd2, 2)):
        dev = max(abs(Phi(ph, j) - fLj(len(ph), j, g)) for j in range(1, n + 1))
        print(
            f"    gamma = {gam:5s} {nm}: max|Phi_L^j - f_L^j| (j <= n) = {mp.nstr(dev, 4):>12s}; "
            f"at j = n+1 it FAILS by {mp.nstr(abs(Phi(ph, n + 1) - fLj(len(ph), n + 1, g)), 6)}  "
            f"[phi_1 = {mp.nstr(ph[0], 8)}]"
        )
print(
    "  theta_0 = pi toggling: PD2 at gamma = 1 through psi_k must give Wimperis BB1 at theta = pi"
)
q = mp.acos(-mp.mpf(1) / 4)
pd2 = [q, -q, -q, q]
psi = []
for k in range(4):
    v = -mp.fsum((-1) ** (h + 1) * pd2[h] for h in range(k)) + mp.fsum(
        (-1) ** (h + 1) * pd2[h] for h in range(k + 1, 4)
    )
    psi.append(v)
print(
    f"    psi = ({', '.join(mp.nstr(x, 8) for x in psi)})  against (phi_1, 3 phi_1, 3 phi_1, phi_1) = "
    f"({mp.nstr(q, 8)}, {mp.nstr(3 * q, 8)}, {mp.nstr(3 * q, 8)}, {mp.nstr(q, 8)})"
)
tog = [(PI, 0)] + [(PI, p) for p in psi[::-1]]
untog = [(PI, 0)] + [(PI, p) for p in pd2[::-1]]
for nm, s in (("toggled (broadband)", tog), ("untoggled (narrowband)", untog)):
    v = [specnorm(fold(s, ea=mp.mpf(x)) - M(PI, 0)) for x in ("1e-2", "1e-3")]
    print(
        f"    {nm:24s} D = {mp.nstr(v[0], 6):>12s}, {mp.nstr(v[1], 6):>12s}  slope "
        f"{mp.nstr(mp.log(v[0] / v[1]) / mp.log(mp.mpf(10)), 5)}"
    )


# --------------------------------------------------------------------------------------------
# 9. toggling-frame control matrix and the amplitude filter function
# --------------------------------------------------------------------------------------------
def Lam(U):
    """Lambda_ij = (1/2) Tr[U^dag sigma_i U sigma_j]; the adjoint of U^dag, an element of SO(3)."""
    S = (SX, SY, SZ)
    L = mp.matrix(3, 3)
    for i in range(3):
        A = U.H * S[i] * U
        for j in range(3):
            B = A * S[j]
            L[i, j] = ((B[0, 0] + B[1, 1]) / 2).real
    return L


def amp_ff_data(seq):
    """per segment: (theta_l, t_{l-1}, t_l, rho-tilde^(l) = rho(phi_l) Lambda^(l-1)); Omega = 1."""
    out, t, Rp = [], mp.mpf(0), I2.copy()
    for thl, phl in seq:
        rt = mp.matrix([[mp.cos(phl), mp.sin(phl), 0]]) * Lam(Rp)
        out.append((thl, t, t + thl, [rt[0, 0], rt[0, 1], rt[0, 2]]))
        t += thl
        Rp = R(thl, phl) * Rp
    return out


def dc_polygon(seq):
    s = [mp.mpf(0)] * 3
    for thl, _, _, rt in amp_ff_data(seq):
        for i in range(3):
            s[i] += thl * rt[i]
    return s


def F_a(seq, w):
    A, B = [mp.mpf(0)] * 3, [mp.mpf(0)] * 3
    for _, t0, t1, rt in amp_ff_data(seq):
        al = mp.cos(w * t1) - mp.cos(w * t0)
        bl = mp.sin(w * t1) - mp.sin(w * t0)
        for i in range(3):
            A[i] += al * rt[i]
            B[i] += bl * rt[i]
    return (mp.fsum(a * a for a in A) + mp.fsum(b * b for b in B)) / 4


head("9. Toggling-frame control matrix R_ij(t) = (1/2)Tr[U_c^dag sigma_i U_c sigma_j]")
for t in ("0.3", "1.1", "2.7"):
    t = mp.mpf(t)
    L = Lam(R(t, 0))
    err = max(abs((L * L.T - mp.eye(3))[i, j]) for i in range(3) for j in range(3))
    print(
        f"  Omega t = {float(t):4.1f}: ||R R^T - I|| = {mp.nstr(err, 4):>10s}, "
        f"R_zx = {mp.nstr(L[2, 0], 6)}, R_zy = {mp.nstr(L[2, 1], 10)} (= sin Omega t = "
        f"{mp.nstr(mp.sin(t), 10)}), R_zz = {mp.nstr(L[2, 2], 10)} (= cos Omega t)"
    )
print(
    "  per-pulse frequency-domain control vector: corrected Green Eq. (40) (explicit Tr/2) against"
)
print(
    "  direct quadrature of -i omega int_0^T dt e^{i omega t} R^{P}_{zj}(t), and the printed 2x form:"
)


def RzP_quad(theta, phi, T, w):
    Om = theta / T
    out = []
    for j, S in enumerate((SX, SY, SZ)):

        def g(t, S=S):
            U = R(Om * t, phi)
            B = U.H * SZ * U * S
            return mp.e ** (1j * w * t) * (B[0, 0] + B[1, 1]) / 2

        out.append(-1j * w * mp.quad(g, [0, T]))
    return out


def RzP_closed(theta, phi, T, w, printed=False):
    Om = theta / T
    f = mp.e ** (1j * w * T) * mp.cos(theta) - 1
    g = mp.e ** (1j * w * T) * mp.sin(theta)
    d = w**2 - Om**2
    tf = 2 if printed else 1  # the printed Tr term is a factor 2 too large
    return [
        tf * 1j * w * mp.sin(phi) / d * (Om * f - 1j * w * g),
        -tf * 1j * w * mp.cos(phi) / d * (Om * f - 1j * w * g),
        w * (1j * Om * g - w * f) / d,
    ]


for theta, phi, T, w in (
    (PI / 2, PI / 2, mp.mpf(1), mp.mpf("0.4")),
    (mp.mpf(2), mp.mpf("0.7"), mp.mpf("1.3"), mp.mpf("2.3")),
    (PI, mp.mpf(0), mp.mpf(1), mp.mpf("2.9")),
):
    q = RzP_quad(theta, phi, T, w)
    c = RzP_closed(theta, phi, T, w)
    p = RzP_closed(theta, phi, T, w, printed=True)
    print(
        f"    (theta,phi,T,w) = ({mp.nstr(theta, 4)},{mp.nstr(phi, 4)},{mp.nstr(T, 3)},{mp.nstr(w, 3)}): "
        f"max|closed - quad| = {mp.nstr(max(abs(a - b) for a, b in zip(c, q)), 4)};  printed/exact ratio "
        f"= ({', '.join(mp.nstr(abs(pp / cc), 6) if abs(cc) > 1e-30 else 'n/a' for pp, cc in zip(p, c))})"
    )
Om = PI
print(f"  primitive pi_X (Green Eqs. 46a/46b, Omega = pi/tau_pi, tau_pi = 1):")
for w in ("0.4", "1.9", "5.5"):
    w = mp.mpf(w)
    c = RzP_closed(PI, mp.mpf(0), mp.mpf(1), w)
    zz = w**2 / (w**2 - Om**2) * (mp.e ** (1j * w) + 1)
    zy = 1j * w * Om / (w**2 - Om**2) * (mp.e ** (1j * w) + 1)
    print(
        f"    w = {float(w):4.1f}: R_zz = {mp.nstr(c[2], 10)} (46a {mp.nstr(zz, 10)}), "
        f"R_zy = {mp.nstr(c[1], 10)} (46b {mp.nstr(zy, 10)}), R_zx = {mp.nstr(abs(c[0]), 4)}"
    )
print(
    f"  free evolution / pure z rotation (theta_l = 0 AND Omega_l = 0 together), T = 1.4, w = 0.5: "
    f"R_z = (0, 0, {mp.nstr(1 - mp.e ** (1j * mp.mpf('0.5') * mp.mpf('1.4')), 12)})"
)

head(
    "10. Amplitude filter function: dc closed polygon, exact primitive form, roll-off, crossover"
)
print(
    "  dc first-order cancellation sum_l theta_l rho-tilde_a^(l) = 0 (a closed polygon), theta = pi:"
)
for name, sf in LIB:
    s = dc_polygon(sf(th))
    print(
        f"    {name:13s} ({mp.nstr(s[0], 6):>12s}, {mp.nstr(s[1], 6):>12s}, {mp.nstr(s[2], 6):>12s})"
        f"   |.| = {mp.nstr(mp.sqrt(mp.fsum(x * x for x in s)), 6)}"
    )
print("  primitive amplitude FF is exactly sin^2(omega tau_P/2):")
for wt in ("0.3", "0.7", "1.9"):
    w = mp.mpf(wt) * PI / th
    print(
        f"    omega tau_P = {wt} pi: F_a = {mp.nstr(F_a([(th, 0)], w), 12)}   "
        f"sin^2 = {mp.nstr(mp.sin(w * th / 2) ** 2, 12)}"
    )
print(
    "  low-frequency log-log slopes of F_a between omega = 1e-5 and 1e-4 Omega (theta = pi):"
)
for name, sf in LIB:
    s = sf(th)
    f1, f2 = F_a(s, mp.mpf("1e-5")), F_a(s, mp.mpf("1e-4"))
    print(
        f"    {name:13s} slope {mp.nstr(mp.log(f2 / f1) / mp.log(mp.mpf(10)), 6):>8s}   "
        f"F_a(1e-4 Omega) = {mp.nstr(f2, 6)}"
    )
tauP, tauCP = th, 4 * PI + th
print(
    f"  printed bound (1/16)(omega tau_CP)^4 at omega = 1e-4: "
    f"{mp.nstr((mp.mpf('1e-4') * tauCP) ** 4 / 16, 6)} -- an UPPER BOUND, ~7x above SK1 and ~25x above BB1"
)
print(
    f"  bound crossover 2 tau_P/tau_CP^2 = {mp.nstr(2 * tauP / tauCP**2, 6)} Omega (the paper's 0.025 Omega); "
    f"true crossovers: SK1 "
    f"{mp.nstr(mp.findroot(lambda w: F_a(seq_sk1(th), w) - mp.sin(w * th / 2) ** 2, mp.mpf('0.1')), 6)} Omega, "
    f"BB1 {mp.nstr(mp.findroot(lambda w: F_a(seq_bb1(th), w) - mp.sin(w * th / 2) ** 2, mp.mpf('0.1')), 6)} Omega"
)

head(
    "11. dc frozen-noise floor: the Omega^{-2(m+1)} normalization and the alternatives it excludes"
)
Omg = mp.mpf("1.5e6")
Pone = mp.mpf("2.07e9")
b2 = Pone / PI
rel = b2 / Omg**2
print(
    f"  benchmark spectrum: one-sided power {mp.nstr(Pone, 4)} (rad/s)^2 held fixed by the A_mu "
    f"normalizer; <beta^2> = (1/pi) x power = {mp.nstr(b2, 8)} (rad/s)^2 at Omega = {mp.nstr(Omg, 3)} rad/s"
)
print(
    f"  dimensionless (beta/Omega)^2 = {mp.nstr(rel, 8)}; Gaussian moments <beta^{{2(m+1)}}> "
    f"= (2m+1)!! <beta^2>^{{m+1}} (3!! = 3, 5!! = 15)"
)
for name, c, m, quoted in (
    ("SK1", mp.mpf("22.8302557111"), 1, "5.86e-6"),
    ("BB1", mp.mpf("9.388566343"), 2, "3.9e-9"),
    ("CORPSE", mp.mpf("0.006500751892"), 1, "3.0e-9"),
):
    dfact = mp.mpf(1)
    for i in range(1, 2 * m + 2, 2):
        dfact *= i
    print(
        f"    {name:7s} c-hat_{m + 1} = {mp.nstr(c, 12):>16s}, floor = c-hat x {int(dfact)}!! x rel^{m + 1} "
        f"= {mp.nstr(c * dfact * rel ** (m + 1), 6)}   (paper {quoted})"
    )
print("  excluded normalizations for the SK1 floor:")
for nm, bb in (
    ("one-sided with 1/(2 pi)", Pone / (2 * PI)),
    ("two-sided without 1/(2 pi)", 2 * Pone),
    ("one-sided without 1/(2 pi)", Pone),
):
    print(
        f"    {nm:28s} {mp.nstr(mp.mpf('22.8302557111') * 3 * (bb / Omg**2) ** 2, 6)}"
    )
print(
    f"    Omega^-4 dropped entirely    {mp.nstr(mp.mpf('22.8302557111') * 3 * b2**2, 6)}  "
    f"[the brief quotes 2.9e20 for this; it does NOT reproduce]"
)


# --------------------------------------------------------------------------------------------
# 12. dephasing filter functions, chi normalization, CP and UDD orders
# --------------------------------------------------------------------------------------------
def deltas_udd(n):
    return [mp.sin(PI * (j + 1) / (2 * n + 2)) ** 2 for j in range(n)]


def deltas_cp(n):
    return [mp.mpf(2 * (j + 1) - 1) / (2 * n) for j in range(n)]


def y_biercuk(x, ds, dpi=0):
    """Biercuk Eq. (2) amplitude, valid for BOTH pulse-number parities; x = omega tau."""
    n = len(ds)
    acc = mp.fsum((-1) ** j * mp.e ** (1j * d * x) for j, d in enumerate(ds, start=1))
    return (
        1 + (-1) ** (n + 1) * mp.e ** (1j * x) + 2 * mp.cos(x * mp.mpf(dpi) / 2) * acc
    )


def Rzz_green45(x, ds):
    """Green Eq. (45): bang-bang, EVEN n only (it presumes Lambda^(n) = I)."""
    acc = mp.fsum((-1) ** l * mp.e ** (1j * d * x) for l, d in enumerate(ds, start=1))
    return 1 - mp.e ** (1j * x) + 2 * acc


head(
    "12. END-TO-END dephasing normalization on free precession (the load-bearing regression)"
)
tau, sig, dbz = mp.mpf(1), mp.mpf(2), mp.mpf("0.3")
S_b = lambda w: mp.sqrt(2 * PI) * dbz**2 / sig * mp.e ** (-(w**2) / (2 * sig**2))
Cor = lambda t: dbz**2 * mp.e ** (-(t**2) * sig**2 / 2)
a1sq = (
    mp.quad(lambda w: S_b(w) * abs(y_biercuk(w * tau, [])) ** 2 / w**2, [0, mp.inf])
    / PI
)
dbl = mp.quad(lambda t1: mp.quad(lambda t2: Cor(t1 - t2), [0, tau]), [0, tau])
print(
    f"  Gaussian S_b(omega) = sqrt(2 pi) (dbz^2/sigma) exp[-omega^2/(2 sigma^2)] with sigma = 2.0 and"
)
print(
    f"  dbz = rms of the sigma_z COEFFICIENT = 0.3; tau = 1; F = 4 sin^2(omega tau/2)"
)
print(
    f"    C(0) = (1/2 pi) int S_b d omega  = {mp.nstr(mp.quad(S_b, [-mp.inf, mp.inf]) / (2 * PI), 12)}  "
    f"(= dbz^2 = 0.09)"
)
print(f"    <a_1^2> = (1/pi) int_0^inf S_b F/omega^2 = {mp.nstr(a1sq, 13)}")
print(
    f"    direct double integral of C(t1-t2) = {mp.nstr(dbl, 13)}   (agreement "
    f"{mp.nstr(abs(a1sq - dbl), 3)})"
)
print(
    f"    W = exp(-2<a_1^2>) = {mp.nstr(mp.e ** (-2 * a1sq), 12)}   "
    f"[the printed exp(-<a_1^2>) = {mp.nstr(mp.e ** (-a1sq), 12)} UNDER-predicts the exponent by 2x]"
)
print(
    f"    F_av = (1 + W)/2 = {mp.nstr((1 + mp.e ** (-2 * a1sq)) / 2, 12)};  1 - F_av = "
    f"{mp.nstr((1 - mp.e ** (-2 * a1sq)) / 2, 12)} against the linear <a_1^2> = {mp.nstr(a1sq, 12)}"
)
print(
    f"    Biercuk chi = (2/pi) int_0^inf S_b F/omega^2 = {mp.nstr(2 * a1sq, 12)} = 2<a_1^2> exactly"
)
print(
    f"    reading S_beta as the SPLITTING-fluctuation PSD (4 S_b) instead: "
    f"{mp.nstr(8 * a1sq, 12)}, a factor 4 too large"
)
print(
    f"    matched-pair check: |int_0^tau rho e^{{i omega t}} dt|^2 = F/omega^2 is finite as omega -> 0: "
    f"F/omega^2 at omega = 1e-8 is {mp.nstr(abs(y_biercuk(mp.mpf('1e-8'), [])) ** 2 / mp.mpf('1e-8') ** 2, 10)} "
    f"(-> tau^2 = 1)"
)

head(
    "13. Filter-function limits, and Biercuk's (-1)^{n+1} form against Green's even-n-only Eq. 45"
)
for x in ("0.3", "1.7", "5.0"):
    x = mp.mpf(x)
    print(
        f"  n = 0 (FID), x = {float(x):4.1f}: F = {mp.nstr(abs(y_biercuk(x, [])) ** 2, 12):>18s}   "
        f"4 sin^2(x/2) = {mp.nstr(4 * mp.sin(x / 2) ** 2, 12)}"
    )
for x in ("0.3", "1.7", "5.0"):
    x = mp.mpf(x)
    print(
        f"  n = 1 (Hahn), x = {float(x):4.1f}: F = {mp.nstr(abs(y_biercuk(x, [mp.mpf(1) / 2])) ** 2, 12):>18s}   "
        f"16 sin^4(x/4) = {mp.nstr(16 * mp.sin(x / 4) ** 4, 12)}"
    )
print("  CP timings at x = 0.9:")
for n in (1, 2, 3, 4):
    ds = deltas_cp(n)
    x = mp.mpf("0.9")
    print(
        f"    n = {n}: Biercuk {mp.nstr(abs(y_biercuk(x, ds)) ** 2, 12):>18s}   "
        f"Green Eq.45 {mp.nstr(abs(Rzz_green45(x, ds)) ** 2, 12):>18s}"
        + ("   (agree, even n)" if n % 2 == 0 else "   (Green Eq.45 invalid at odd n)")
    )
print(
    f"  Green Eq.45 at omega -> 0 and odd n gives |R_zz|^2 = "
    f"{[mp.nstr(abs(Rzz_green45(mp.mpf('1e-40'), deltas_cp(n))) ** 2, 6) for n in (1, 3, 5)]} "
    f"-- no low-frequency suppression at all"
)
print(
    f"  annihilation point omega tau_pi = pi (n = 4, delta_pi = 0.07): F = "
    f"{mp.nstr(abs(y_biercuk(PI / mp.mpf('0.07'), deltas_cp(4), mp.mpf('0.07'))) ** 2, 12)} = "
    f"4 sin^2(omega tau/2) = {mp.nstr(4 * mp.sin(PI / mp.mpf('0.07') / 2) ** 2, 12)} (the FID form)"
)
print(f"  UDD and CPMG timings coincide at n = 1 and n = 2 and first differ at n = 3:")
for n in (1, 2, 3):
    print(
        f"    n = {n}: UDD {[mp.nstr(d, 10) for d in deltas_udd(n)]}  CPMG {[mp.nstr(d, 10) for d in deltas_cp(n)]}"
    )

head(
    "14. Low-frequency suppression orders at high precision (mpmath; double precision fails here)"
)


def local_order(fn, n, extra=60, dec=8):
    """local log-log exponent at omega tau -> 0 with enough digits for the cancellation."""
    old = mp.mp.dps
    mp.mp.dps = dec * (2 * n + 6) + extra
    x = mp.mpf(10) ** (-dec)
    v1, v2 = fn(x), fn(x / 10)
    o = mp.log(abs(v1) / abs(v2)) / mp.log(mp.mpf(10))
    mp.mp.dps = old
    return o, v1, x


print(
    "  Carr-Purcell delta_l = (l - 1/2)/n, bang-bang: lowest nonvanishing Taylor order of R_zz"
)
for n in (2, 4, 6, 8):
    o, v1, x = local_order(lambda x, n=n: Rzz_green45(x, deltas_cp(n)), n)
    lead = mp.mpc(v1 / x**3)
    print(
        f"    n = {n}: order {mp.nstr(o, 8):>10s} (so F_z ~ omega^6, alpha = 2, 18 dB/octave); "
        f"leading x^3 coefficient {mp.nstr(lead.imag, 10)} i"
    )
print(
    "  Uhrig delta_l = sin^2[pi l/(2n+2)], bang-bang: order n+1, so F_z ~ omega^{2(n+1)}, alpha = n"
)
for n in (2, 3, 4, 5, 6, 8, 10, 12):
    o, _, _ = local_order(lambda x, n=n: y_biercuk(x, deltas_udd(n)), n)
    print(f"    n = {n:2d}: order {mp.nstr(o, 8):>10s}   expected {n + 1}")
print(
    "  ideal-pulse leading coefficients (F -> (omega tau)^{2n+2}/(16^n (n!)^2) for UDD):"
)
old = mp.mp.dps
mp.mp.dps = 400
xs = mp.mpf(10) ** (-25)
for n in (1, 2, 3, 4, 5, 6, 7, 8):
    v = abs(y_biercuk(xs, deltas_udd(n))) ** 2 / xs ** (2 * n + 2)
    cf = mp.mpf(1) / (mp.mpf(16) ** n * mp.factorial(n) ** 2)
    print(
        f"    UDD  n = {n}: {mp.nstr(v, 12):>20s}   closed form {mp.nstr(cf, 12):>20s}"
    )
for n in (1, 3, 5, 7):
    v = abs(y_biercuk(xs, deltas_cp(n))) ** 2 / xs**4
    print(
        f"    CPMG n = {n} (odd):  F/(omega tau)^4 = {mp.nstr(v, 12):>18s}   "
        f"1/(16 n^4) = {mp.nstr(mp.mpf(1) / (16 * mp.mpf(n) ** 4), 12)}"
    )
for n in (2, 4, 6, 8):
    v = abs(y_biercuk(xs, deltas_cp(n))) ** 2 / xs**6
    print(
        f"    CPMG n = {n} (even): F/(omega tau)^6 = {mp.nstr(v, 12):>18s}   "
        f"1/(64 n^4) = {mp.nstr(mp.mpf(1) / (64 * mp.mpf(n) ** 4), 12)}"
    )
mp.mp.dps = old
print("  FINITE-PULSE COLLAPSE of the UDD order at delta_pi = tau_pi/tau = 0.02:")
for n in (1, 2, 3, 4, 5, 6, 12):
    oi, _, _ = local_order(lambda x, n=n: abs(y_biercuk(x, deltas_udd(n))) ** 2, n)
    of, v1, x = local_order(
        lambda x, n=n: abs(y_biercuk(x, deltas_udd(n), mp.mpf("0.02"))) ** 2, n
    )
    extra = ""
    if n >= 3 and n % 2 == 1:
        extra = f"   F/(omega tau_pi)^4 = {mp.nstr(v1 / (x * mp.mpf('0.02')) ** 4, 10)} (= 1/16, universal in n)"
    elif n >= 4 and n % 2 == 0:
        extra = f"   F/[(omega tau)^2 (omega tau_pi)^4] = {mp.nstr(v1 / (x**2 * (x * mp.mpf('0.02')) ** 4), 10)} (= 1/64)"
    print(
        f"    n = {n:2d}: F exponent {mp.nstr(oi, 6):>8s} ideal -> {mp.nstr(of, 6):>8s} at "
        f"delta_pi = 0.02{extra}"
    )
print(
    "  echo moment conditions A_k = sum_j (-1)^j delta_j^k, satisfied by BOTH families:"
)
for n in (3, 4, 5, 6):
    for nm, ds in (("UDD ", deltas_udd(n)), ("CPMG", deltas_cp(n))):
        A0 = mp.fsum((-1) ** j for j in range(1, n + 1))
        A1 = mp.fsum((-1) ** j * d for j, d in enumerate(ds, start=1))
        A2 = mp.fsum((-1) ** j * d**2 for j, d in enumerate(ds, start=1))
        print(
            f"    n = {n} {nm}: A_0 = {mp.nstr(A0, 3):>5s}, A_1 = {mp.nstr(A1, 6):>10s}, "
            f"A_2 = {mp.nstr(A2, 8):>12s}"
        )

head(
    "15. IR convergence of chi on an ambient-like 1/omega^4 spectrum (tau = 1, tau_pi = 0.05)"
)
for n in (3, 4, 5, 6):
    ds = deltas_udd(n)
    vals = []
    for wmin in ("1e-7", "1e-8"):
        wmin = mp.mpf(wmin)
        vals.append(
            2
            / PI
            * mp.quad(
                lambda w: abs(y_biercuk(w, ds, mp.mpf("0.05"))) ** 2 / w**6,
                [wmin, mp.mpf(1), mp.mpf(50)],
            )
        )
    print(
        f"  UDD n = {n}: chi(omega_min = 1e-7) = {mp.nstr(vals[0], 8):>14s}, "
        f"chi(1e-8) = {mp.nstr(vals[1], 8):>14s}, ratio {mp.nstr(vals[1] / vals[0], 8)}"
        + (
            "   (odd n: chi proportional to 1/omega_min, IR DIVERGENT)"
            if n % 2
            else "   (even n: IR convergent)"
        )
    )
print(
    "  any chi quoted on a 1/omega^4 spectrum at odd n is meaningless without omega_min stated."
)

head("16. Durations at constant amplitude: sum_l theta_l against the closed forms of PLAN.md Section 4.3.5")
print("   (2026-09-04 critique: the concatenations are 8 pi + theta - 4k, not 8 pi + 2 theta - 4k)")
for thd in ("0.5", "1.0", "1.5"):
    tht = mp.mpf(thd) * PI
    kk = mp.asin(mp.sin(tht / 2) / 2)
    forms = {
        "SK1": 4 * PI + tht,
        "BB1 = B2": 4 * PI + tht,
        "CORPSE": 4 * PI + tht - 4 * kk,
        "red. CinSK": 8 * PI + tht - 4 * kk,
        "red. CinBB": 8 * PI + tht - 4 * kk,
    }
    seqs = {
        "SK1": seq_sk1(tht),
        "BB1 = B2": seq_bb1(tht),
        "CORPSE": seq_corpse(tht),
        "red. CinSK": seq_cinsk(tht),
        "red. CinBB": seq_cinbb(tht),
    }
    for name, s in seqs.items():
        tot = sum(x[0] for x in s)
        assert abs(tot - forms[name]) < mp.mpf("1e-40"), (name, thd, tot, forms[name])
        print(
            f"  theta = {thd} pi {name:12s}: sum theta_l = {mp.nstr(tot / PI, 8):>10s} pi = closed form {mp.nstr(forms[name] / PI, 8):>10s} pi"
            + (f"   (printed 8 pi + 2 theta - 4k = {mp.nstr((8 * PI + 2 * tht - 4 * kk) / PI, 8)} pi, WRONG)" if name.startswith("red.") else "")
        )

print()
print("done.")
