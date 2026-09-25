"""9Be+ species table (PLAN.md Section 4.5.6): I = 3/2, mu_I < 0, so F = 1 lies above F = 2.

The ground-state A and g_J are Wineland, Bollinger and Itano 1983's measurements (the g_J re-reduced by Shiga et al.
2011, where the digits are printed), mu_I is Dickopf et al. 2024's corrected moment, A(2p 2P1/2) Noertershaeuser et al.
2009's. No 2p 2P3/2 hyperfine constant has been measured: the theory values (Puchalski and Pachucki 2009) exceed Bollinger
et al. 1985's bound |A| < 0.6 MHz. The P linewidth spans 9% across sources (17.97, 19.4, 19.6, 19.64 MHz); the record uses
Monroe's 19.4 MHz, which the clock-point anchors assume. The 2p g_J are Lande values ([background]).
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    CitedFactory,
    energy_hz,
    ion_mass_u,
    lifetime_s_from_linewidth,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "9Be+"
_c = CitedFactory("be9.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 9.012183065, "u", "NIST_AWIC", uncertainty=8.2e-8),
    _c("nuclear_spin", 1.5, "hbar", "Ozeri2007", tag="verified", note="Ozeri Table I; Langer 2005"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.177432,
        "mu_N",
        "Dickopf2024",
        tag="verified",
        uncertainty=5e-6,
        note="the diamagnetically CORRECTED (bare-nucleus) moment, g_I = -0.78495442296(42)(11); the uncorrected NMR "
        "value -1.17449(2) mu_N is a different quantity, and 25Mg+'s table carries the uncorrected convention",
    ),
    _c(
        "S12.A_hfs_hz",
        -625.008837048e6,
        "Hz",
        "WinelandBollingerItano1983",
        tag="verified",
        uncertainty=1e-2,
        note="SIGNED negative (inverted multiplet), printed as 'preliminary'; Shiga et al. 2011's zero-field A_0 is 4 "
        "mHz away and adds A(B) = A_0(1 + k B^2), a field dependence the Breit-Rabi solve does not model",
    ),
    _c(
        "S12.A_hfs_shiga_hz",
        -625.008837044e6,
        "Hz",
        "Shiga2011",
        tag="verified",
        uncertainty=1.2e-2,
        note="the modern zero-field value; a cross-check",
    ),
    _c(
        "S12.g_J",
        2.00226239,
        "",
        "Shiga2011",
        tag="corrected",
        uncertainty=3.1e-7,
        note="Wineland, Bollinger and Itano 1983's cyclotron versus hyperfine-Zeeman measurement, re-reduced with "
        "CODATA 2006 and printed in the body of Shiga et al. 2011 (their abstract gives only the ratio g_I'/g_J); "
        "Langer's thesis prints the negative in the other g-factor convention",
    ),
    _c(
        "S12.g_J_theory",
        2.0022621287,
        "",
        "Dickopf2024",
        tag="background",
        uncertainty=2.4e-10,
        note="a calculation (printed negative in their convention), 0.84 sigma below the measurement; a cross-check "
        "that moves the clock point by 0.0093 Hz",
    ),
    _c(
        "P12.A_hfs_hz",
        -118.00e6,
        "Hz",
        "Nortershauser2009",
        tag="verified",
        uncertainty=0.04e6,
        note="Table II; 90x tighter than Bollinger et al. 1985's -118.6(3.6) MHz",
    ),
    _c(
        "P12.A_hfs_bollinger_hz",
        -118.6e6,
        "Hz",
        "Bollinger1985",
        tag="verified",
        uncertainty=3.6e6,
        note="the first measurement; a cross-check",
    ),
    _c(
        "P32.A_hfs_hz",
        -1.026e6,
        "Hz",
        "PuchalskiPachucki2009",
        tag="background",
        uncertainty=0.003e6,
        note="THEORY, and above Bollinger et al. 1985's experimental bound |A| < 0.6 MHz; nil effect on the anchors (a "
        "1 MHz splitting against THz detunings)",
    ),
    _c(
        "P32.B_hfs_hz",
        -2.29940e6,
        "Hz",
        "PuchalskiPachucki2009",
        tag="background",
        uncertainty=0.00003e6,
        note="THEORY; its sign is questionable, since for one p3/2 electron sign(B) = sign(Q) and Q(9Be) > 0",
    ),
    _c(
        "P32.A_hfs_bound_hz",
        0.6e6,
        "Hz",
        "Bollinger1985",
        tag="contested",
        note="the experimental upper bound on |A(2p 2P3/2)| (quoted from Poulsen et al.)",
    ),
    _c(
        "P.linewidth_nist_hz",
        17.97e6,
        "Hz",
        "NIST_ASD_5_12",
        tag="contested",
        note="from the NIST ASD A_ki (grade AAA), traced to Yan, Tambasco and Drake 1998 (theory): tau = 8.86 ns, f = "
        "0.498; the trapped-ion 19.4 / 19.6 MHz need f = 0.543",
    ),
    _c(
        "P32.fine_structure_splitting_hz",
        197_063.48e6,
        "Hz",
        "Nortershauser2009",
        tag="verified",
        uncertainty=0.52e6,
        note="measured; the NIST level difference (197.14 GHz) is 80 MHz off, so its wavelengths are right to 4e-4",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        0.0529,
        "b",
        "Stone2019",
        tag="verified",
        uncertainty=0.0004,
        note="Puchalski, Komasa and Pachucki 2021 give +0.05350(14) b",
    ),
    _c("P12.energy_cm", 31928.744, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c("P32.energy_cm", 31935.320, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c("P.linewidth_ozeri_hz", 19.6e6, "Hz", "Ozeri2007", tag="verified", note="Ozeri Table I gamma/2pi"),
    _c(
        "P.linewidth_monroe_hz",
        19.4e6,
        "Hz",
        "Monroe1995",
        tag="verified",
        note="Gamma/2pi of the 313 nm cycling line as PLAN.md 4.2.1 quotes it; the value the record uses",
    ),
    _c(
        "S12.hfs_splitting_ozeri_hz",
        1.25e9,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I, rounded; |A|(I + 1/2) = 1.250018 GHz",
    ),
    _c(
        "fine_structure_splitting_hz",
        0.198e12,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_f/2pi; the NIST levels give 197.1 GHz",
    ),
    _c(
        "wavelength_ozeri_D1_m",
        313.1e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR; NIST vacuum 313.197 nm",
    ),
    _c(
        "wavelength_ozeri_D2_m",
        313.0e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR; NIST vacuum 313.133 nm",
    ),
    _c("P_to_D_branching", 0.0, "", "Ozeri2007", tag="verified", note="no D level below the P levels"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """The ``Species`` record for 9Be+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    e_p12 = energy_hz(t["be9.P12.energy_cm"])
    e_p32 = energy_hz(t["be9.P32.energy_cm"])
    tau_p = lifetime_s_from_linewidth(t["be9.P.linewidth_monroe_hz"])
    gamma = t["be9.P.linewidth_monroe_hz"].value
    s12 = Level(
        "S1/2",
        0.0,
        None,
        t["be9.S12.A_hfs_hz"].value,
        0.0,
        t["be9.S12.g_J"].value,
        ("WinelandBollingerItano1983", "Shiga2011", "Dickopf2024"),
    )
    p12 = Level(
        "P1/2",
        e_p12,
        tau_p,
        t["be9.P12.A_hfs_hz"].value,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "Nortershauser2009", "Monroe1995", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        tau_p,
        t["be9.P32.A_hfs_hz"].value,
        t["be9.P32.B_hfs_hz"].value,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "PuchalskiPachucki2009", "Monroe1995", "PLAN_background"),
    )
    # no D level lies below the 2p levels, so each P level decays only to S1/2
    cites = ("NIST_ASD_5_12", "Monroe1995", "Ozeri2007")
    transitions = (
        Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), gamma, 1.0, "E1", cites),
        Transition("S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), gamma, 1.0, "E1", cites),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["be9.mass_atomic_u"]),
        nuclear_spin=float(Fraction(t["be9.nuclear_spin"].value).limit_denominator(2)),
        mu_I_nuclear_magnetons=t["be9.mu_I_nuclear_magnetons"].value,
        levels=(s12, p12, p32),
        transitions=transitions,
        # the |2,0> <-> |1,+1> clock qubit at 119.446 G and 313 nm sigma+ cycling through P3/2; no repump, no shelf
        qubit=("S1/2 F=2 mF=0", "S1/2 F=1 mF=1"),
        cycling="S1/2-P3/2",
        repumps=(),
        shelving=None,
    )
