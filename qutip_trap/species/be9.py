"""9Be+ species table (I = 3/2).

The 2p 2P3/2 hyperfine constants are theory only (Puchalski and Pachucki 2009), and the theory A exceeds the
experimental bound |A| < 0.6 MHz. The P linewidth readings span 9% (17.97 to 19.64 MHz); the record uses Monroe et
al. 1995's 19.4 MHz.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    Cited,
    IncompleteSpeciesTable,
    MissingConstant,
    energy_hz,
    ion_mass_u,
    lifetime_s_from_linewidth,
    required_constants_missing,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "9Be+"
_c = cited_factory("be9.")

_ENTRIES: tuple[Cited, ...] = (
    _c(
        "mass_atomic_u",
        9.012183065,
        "u",
        "NIST_AWIC",
        uncertainty=8.2e-8,
        note="PLAN.md 4.1.7 fixtures use 9.0121822 u for the Home 2013 Be+/Mg+ crystal; the difference is 1e-7 relative",
    ),
    _c("nuclear_spin", 1.5, "hbar", "Ozeri2007", tag="verified", note="Ozeri Table I; Langer 2005"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.177432,
        "mu_N",
        "Dickopf2024",
        tag="verified",
        uncertainty=5e-6,
        note="the DIAMAGNETICALLY CORRECTED (bare-nucleus) moment; Dickopf et al. 2024's "
        "g_I = -0.78495442296(42)(11) gives -1.1774316344 mu_N, which reproduces the value PLAN.md 9.13 and "
        "check_atomic.py use to 3e-7, and Stone INDC(NDS)-0794 recommends -1.177430(5). The UNCORRECTED NMR "
        "value is -1.17449(2) mu_N, a 0.25% different quantity -- so 25Mg+, whose table carries the "
        "uncorrected -0.85545, is in a DIFFERENT shielding convention (Section 13's mu_I row requires the "
        "convention be declared per table). Negative, so F = I - 1/2 = 1 lies above F = 2",
    ),
    _c(
        "S12.A_hfs_hz",
        -625.008837048e6,
        "Hz",
        "WinelandBollingerItano1983",
        tag="verified",
        uncertainty=1e-2,
        note="SIGNED negative (inverted multiplet). Wineland, Bollinger and Itano 1983 label this value "
        "'preliminary'; Shiga, Itano and Bollinger 2011's zero-field A_0 = -625.008837044(12) MHz supersedes "
        "it by 4 mHz (not a conflict), and adds A(B) = A_0(1 + k B^2) with k = 2.63(18)e-11 T^-2, a "
        "field-dependent correction the Breit-Rabi solve does not model",
    ),
    _c(
        "S12.A_hfs_shiga_hz",
        -625.008837044e6,
        "Hz",
        "Shiga2011",
        tag="verified",
        uncertainty=1.2e-2,
        note="the modern zero-field value; cross-check only, 4 mHz from the value used",
    ),
    _c(
        "S12.g_J",
        2.00226239,
        "",
        "Shiga2011",
        tag="corrected",
        uncertainty=3.1e-7,
        note="MEASURED, 50x tighter than the 2.00226(2) PLAN.md 9.13 carries uncited. PROVENANCE RE-CHECKED "
        "2026-09-08 (ledger conv.be9_g_j_provenance) after a literature review claimed the Shiga citation "
        "was wrong on the grounds that the paper's ABSTRACT reports only A_0, k and the RATIO g_I'/g_J. "
        "The claim is false: Shiga et al.'s Sec. I BODY prints the absolute value, read verbatim in the "
        "full text -- 'The value of g_J for the ground electronic state of 9Be+ has been determined by "
        "measuring the 9Be+ cyclotron frequency and a hyperfine-Zeeman transition frequency at the same "
        "magnetic field [5]. The value is g_J = 2.002 262 39(31), calculated with the use of the best "
        "current value of the proton-electron mass ratio [6].' Their [5] is Wineland, Bollinger and Itano "
        "1983 (the MEASUREMENT, whose own printed value is 2.00226206(42)) and their [6] is Mohr, Taylor "
        "and Newell, Rev. Mod. Phys. 80, 633 (2008), i.e. CODATA 2006, from which Shiga et al. re-reduce "
        "the digits. So the chain is: measured 1983, re-evaluated 2011, and Shiga 2011 is where these "
        "digits are printed -- the citation stands and the value is unchanged. Langer's thesis prints the "
        "negative of this in the other g-factor convention; Section 13's g-factor row governs, so the "
        "positive value is stored. This is also the value 25Mg+'s table used to borrow, which it must not "
        "(25Mg+ has no measured g_J). See S12.g_J_theory for the modern CALCULATED value, which does NOT "
        "replace this one",
    ),
    _c(
        "S12.g_J_theory",
        2.0022621287,
        "",
        "Dickopf2024",
        tag="background",
        uncertainty=2.4e-10,
        note="THEORY, and a cross-check only -- it is NOT what the table uses. Dickopf et al. 2024 need a "
        "bound-electron g factor of 9Be+ to extract the diamagnetic shielding, and take it from a "
        "calculation, in their own words: 'the bound-electron g-factor g_s(9Be+) = -2.0022621287(24). For "
        "the latter, we use the calculations performed in ref. [22] and the updated nuclear recoil "
        "correction [33, 34].' Printed NEGATIVE in their convention; the magnitude is stored, as Section "
        "13's g-factor row requires. It is 2.61e-7 BELOW the measured 2.00226239(31), i.e. 0.84 sigma of "
        "the measurement's own bar and 130x more precise, so the two AGREE and nothing here contradicts "
        "the stored value. Swapping it in moves the Section 9.13 clock point by 1.56e-5 G and 0.0093 Hz "
        "(tests/test_species_gaps.py pins both), 2000x inside the 20 Hz tolerance the plan sets on that "
        "anchor -- so the anchor cannot distinguish the two, and there is no reason to replace a "
        "measurement with a calculation. For scale, the plan's own negative control g_J = 2.000 moves the "
        "same point by 0.135 G and 81 Hz",
    ),
    _c(
        "P12.A_hfs_hz",
        -118.00e6,
        "Hz",
        "Nortershauser2009",
        tag="verified",
        uncertainty=0.04e6,
        note="Table II; 90x tighter than Bollinger et al. 1985's first measurement -118.6(3.6) MHz",
    ),
    _c(
        "P12.A_hfs_bollinger_hz",
        -118.6e6,
        "Hz",
        "Bollinger1985",
        tag="verified",
        uncertainty=3.6e6,
        note="the first measurement; cross-check only",
    ),
    _c(
        "P32.A_hfs_hz",
        -1.026e6,
        "Hz",
        "PuchalskiPachucki2009",
        tag="background",
        uncertainty=0.003e6,
        note="THEORY, not a measurement: no 9Be+ 2p 2P3/2 hyperfine constant has been measured, and this value "
        "EXCEEDS Bollinger et al. 1985's experimental bound |A(2P3/2)| < 0.6 MHz. The tension is unresolved "
        "and recorded in the ledger as conv.be9_p32_hyperfine; the effect on any Section 9.13 anchor is nil "
        "(a 1 MHz excited-state splitting against THz detunings), but the value must never be read as "
        "measured. The -1.03(3) MHz that circulates is this number with an invented uncertainty",
    ),
    _c(
        "P32.B_hfs_hz",
        -2.29940e6,
        "Hz",
        "PuchalskiPachucki2009",
        tag="background",
        uncertainty=0.00003e6,
        note="THEORY. Its SIGN is questionable: for a single p3/2 valence electron sign(B) = sign(Q), and "
        "Q(9Be) = +0.0529(4) b is positive (87Rb and 133Cs both obey the rule), so either the paper uses "
        "another B or Q convention or one of the two needs correcting. Recorded in the ledger",
    ),
    _c(
        "P32.A_hfs_bound_hz",
        0.6e6,
        "Hz",
        "Bollinger1985",
        tag="contested",
        note="the EXPERIMENTAL UPPER BOUND |A(2p 2P3/2)| < 0.6 MHz that Bollinger et al. 1985 quote from "
        "Poulsen et al.; the theory value above is 1.7x larger",
    ),
    _c(
        "P.linewidth_nist_hz",
        17.97e6,
        "Hz",
        "NIST_ASD_5_12",
        tag="contested",
        note="Gamma/2pi from the NIST ASD A_ki (1.1292e8 and 1.1285e8 s^-1, accuracy grade AAA, <= 0.3%), "
        "traced to Yan, Tambasco and Drake, Phys. Rev. A 57, 1652 (1998) -- theory. It implies tau = 8.86 ns "
        "and an oscillator strength f = 0.498, the textbook ~0.5 for a Li-like ion, so ASD is "
        "self-consistent; the trapped-ion 19.4 / 19.6 MHz readings need f = 0.543. The four readings span 9%, "
        "far outside every stated accuracy, and no primary derivation of the 19.4 MHz could be located "
        "(ledger conv.be9_linewidth)",
    ),
    _c(
        "P32.fine_structure_splitting_hz",
        197_063.48e6,
        "Hz",
        "Nortershauser2009",
        tag="verified",
        uncertainty=0.52e6,
        note="197 063.48(52) MHz, measured. The NIST ASD level difference (197.14 GHz) is 80 MHz stale, so "
        "the wavelengths derived from those energies are right to 4e-4 but the splitting must not be",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        0.0529,
        "b",
        "Stone2019",
        tag="verified",
        uncertainty=0.0004,
        note="Q(9Be) > 0; Puchalski, Komasa and Pachucki 2021 give +0.05350(14) b",
    ),
    _c("P12.energy_cm", 31928.744, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c("P32.energy_cm", 31935.320, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c(
        "P.linewidth_ozeri_hz",
        19.6e6,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I gamma/2pi; Monroe 1995 quotes 19.4 MHz (PLAN.md 4.2.1), Wineland 2003 19.4 MHz (9.13)",
    ),
    _c(
        "P.linewidth_monroe_hz",
        19.4e6,
        "Hz",
        "Monroe1995",
        tag="verified",
        note="Gamma/2pi of the 313 nm cycling line as PLAN.md 4.2.1 quotes it",
    ),
    _c(
        "S12.hfs_splitting_ozeri_hz",
        1.25e9,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_0/2pi, rounded; |A|(I + 1/2) = 1.250018 GHz",
    ),
    _c(
        "fine_structure_splitting_hz",
        0.198e12,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_f/2pi; the NIST levels give 6.576 cm^-1 = 197.1 GHz",
    ),
    _c(
        "wavelength_ozeri_D1_m",
        313.1e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR wavelength; NIST vacuum 313.197 nm",
    ),
    _c(
        "wavelength_ozeri_D2_m",
        313.0e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR wavelength; NIST vacuum 313.133 nm",
    ),
    _c(
        "P_to_D_branching",
        0.0,
        "",
        "Ozeri2007",
        tag="verified",
        note="no D level below the P levels (Ozeri Table I)",
    ),
)

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

REQUIRED: dict[str, str] = {
    "be9.mass_atomic_u": "relative atomic mass of the neutral atom",
    "be9.mu_I_nuclear_magnetons": "signed nuclear moment, with its shielding convention declared",
    "be9.S12.A_hfs_hz": "ground-state hyperfine A, signed",
    "be9.S12.g_J": "measured ground-state g_J",
    "be9.P12.A_hfs_hz": "2p 2P1/2 hyperfine A",
    "be9.P32.A_hfs_hz": "2p 2P3/2 hyperfine A",
    "be9.P32.B_hfs_hz": "2p 2P3/2 hyperfine B",
    "be9.P12.energy_cm": "2p 2P1/2 level energy",
    "be9.P32.energy_cm": "2p 2P3/2 level energy",
    "be9.P.linewidth_monroe_hz": "the P-level linewidth reading the record adopts",
}
"""The ids ``species()`` needs, each with a description; :data:`MISSING` is derived from it."""

MISSING: tuple[MissingConstant, ...] = required_constants_missing(TABLE, REQUIRED, {})

_OPEN: tuple[MissingConstant, ...] = (
    MissingConstant(
        "a MEASURED 2p 2P3/2 hyperfine A and B",
        "none exists: Puchalski and Pachucki 2009's theory is used and tagged [background], and it exceeds "
        "Bollinger et al. 1985's experimental bound |A| < 0.6 MHz (ledger conv.be9_p32_hyperfine)",
    ),
    MissingConstant(
        "a resolution of the 9% spread in the P linewidth (17.97 / 19.4 / 19.6 / 19.64 MHz)",
        "no primary derivation of the ubiquitous trapped-ion 19.4 MHz could be located; Andersen, Jessen and "
        "Soerensen, Phys. Rev. 188, 76 (1969) is what Ozeri et al. cite. Ledger conv.be9_linewidth",
    ),
    MissingConstant(
        "measured g_J of 2p 2P1/2 and 2p 2P3/2",
        "no measurement found; the Lande values are used and tagged [background]",
    ),
)
"""Declared gaps that do not block the build (the record uses a substitute for each)."""


def species() -> Species:
    """The ``Species`` record for 9Be+, built from :data:`TABLE` alone."""
    if MISSING:
        raise IncompleteSpeciesTable(NAME, MISSING)
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
    # no D level lies below 2p, so each P level decays only to S1/2
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
        # the |2,0> <-> |1,+1> clock qubit (field-independent at 119.446 G); no D level, so no repump or shelf
        qubit=("S1/2 F=2 mF=0", "S1/2 F=1 mF=1"),
        cycling="S1/2-P3/2",
        repumps=(),
        shelving=None,
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
