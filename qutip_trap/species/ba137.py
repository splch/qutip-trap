"""137Ba+ species table (PLAN.md Section 4.5.6): I = 3/2, mu_I > 0, so F = 2 lies above F = 1 in 6s 2S1/2.

The hyperfine constants are Blatt and Werth 1982 (6s), Villemoes et al. 1993 (6p) and Lewty et al. 2013 (5d); lifetimes
and branchings are the Barrett group's. The three measured g_J (6s, 5d 2D3/2, 5d 2D5/2) were taken on 138Ba+ and
135Ba+ and are applied as isotope-independent; the 6p levels carry the Lande values ([background]) and the NIST ASD Lande
column (demonstrably wrong at 5d 2D5/2) is kept only as ``*_g_J_asd`` cross-checks. These are also the isotope-independent
Ba II constants of the 133Ba+ table. PLAN.md prints no clock-point anchor for 137Ba+.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    CitedFactory,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "137Ba+"
_c = CitedFactory("ba137.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 136.90582714, "u", "NIST_AWIC", uncertainty=3.0e-7),
    _c("nuclear_spin", 1.5, "hbar", "BlattWerth1982", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        0.937365,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=2e-5,
        note="positive, unlike 9Be+, 43Ca+ and 133Ba+ (g_I = 0.6250(1)); quoted in Lewty et al. 2012",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        0.236,
        "b",
        "Stone2019",
        tag="contested",
        uncertainty=0.003,
        note="Mertzimekis, Stamou, Psaltis, Nucl. Instrum. Methods A 807, 56 (2016); Lewty et al. 2013 use 0.246(1) b",
    ),
    _c(
        "S12.hfs_splitting_hz",
        8_037_741_667.69,
        "Hz",
        "BlattWerth1982",
        tag="verified",
        uncertainty=3.6e-4,
        note="W_hfs(F = 2 <-> 1)/h, the microwave clock frequency",
    ),
    _c(
        "S12.A_hfs_hz",
        4018.87083385e6,
        "Hz",
        "BlattWerth1982",
        tag="verified",
        uncertainty=1.8e-4,
        note="A = W_hfs/(I + 1/2) as printed; its last digit rounds up from the exact half of the splitting",
    ),
    _c(
        "S12.g_J",
        2.00249192,
        "",
        "Marx1998",
        tag="extracted",
        uncertainty=3e-8,
        note="measured on 138Ba+ and 135Ba+, isotope-independent to the quoted 3e-8 (Hanley et al., arXiv:2105.10352); "
        "the digits are read in Arnold et al., Phys. Rev. Lett. 124, 193001 (2020)",
    ),
    _c("P12.A_hfs_hz", 743.7e6, "Hz", "Villemoes1993", tag="verified", uncertainty=0.3e6),
    _c(
        "P32.A_hfs_hz",
        127.2e6,
        "Hz",
        "Villemoes1993",
        tag="verified",
        uncertainty=0.2e6,
        note="read in one secondary compilation, whose 135Ba+ row matches an independent quotation",
    ),
    _c("P32.B_hfs_hz", 92.5e6, "Hz", "Villemoes1993", tag="verified", uncertainty=0.2e6),
    _c(
        "D32.A_hfs_hz",
        189.731494e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=1.7e-2,
        note="Tables II and VI; supersedes Silverans et al. 1986's 189.7296(7) MHz by about 1000x",
    ),
    _c("D32.B_hfs_hz", 44.537594e6, "Hz", "Lewty2013", tag="verified", uncertainty=3.4e-2),
    _c(
        "D52.A_hfs_hz",
        -12.029234e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=1.1e-2,
        note="NEGATIVE, unlike every other 137Ba+ level",
    ),
    _c("D52.B_hfs_hz", 59.525520e6, "Hz", "Lewty2013", tag="verified", uncertainty=1.10e-1),
    _c(
        "D52.octupole_moment_mu_N_b",
        0.05057,
        "",
        "Lewty2013",
        tag="verified",
        uncertainty=0.00054,
        note="the magnetic octupole moment in mu_N b; the Hamiltonian carries only A and B, so a declared omission",
    ),
    _c(
        "P12.lifetime_s",
        7.855e-9,
        "s",
        "Arnold2019",
        tag="verified",
        uncertainty=0.010e-9,
        note="measured on 138Ba+; gamma/2pi = 20.263 MHz total",
    ),
    _c(
        "P32.lifetime_s",
        6.2615e-9,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.0072e-9,
        note="gamma/2pi = 25.418 MHz total",
    ),
    _c(
        "P12.branching_to_D32",
        0.268177,
        "",
        "Arnold2019",
        tag="verified",
        uncertainty=0.000057,
        note="649.9 nm; De Munshi et al. 2015's 0.2696(4) is 3 sigma away, so the two groups' sets are not mixed",
    ),
    _c(
        "P32.branching_to_D52",
        0.230253,
        "",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.000061,
        note="614.3 nm; with 0.741716(71) into S1/2 and 0.028031(23) into D3/2 the three sum to 1",
    ),
    _c(
        "P32.branching_to_D32",
        0.028031,
        "",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.000023,
        note="585.5 nm",
    ),
    _c(
        "D52.lifetime_s",
        30.14,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.40,
        note="measured on 138Ba+; Auchter et al. 2014's 31.2(9) s agrees at 1.1 sigma",
    ),
    _c(
        "D32.lifetime_s",
        79.8,
        "s",
        "Yu1997",
        tag="verified",
        uncertainty=4.6,
        note="Gurell et al. 2007: 89(16) s",
    ),
    _c("P_to_D_branching_inverse", 3.0, "", "Ozeri2007", tag="verified", note="Ozeri Table I f^-1"),
    _c("D32.energy_cm", 4873.852, "cm^-1", "NIST_ASD_5_12"),
    _c("D52.energy_cm", 5674.807, "cm^-1", "NIST_ASD_5_12"),
    _c("P12.energy_cm", 20261.561, "cm^-1", "NIST_ASD_5_12", note="493.55 nm vacuum"),
    _c("P32.energy_cm", 21952.404, "cm^-1", "NIST_ASD_5_12", note="455.53 nm vacuum"),
    _c(
        "D32.g_J",
        0.7993278,
        "",
        "Knoll1996",
        tag="extracted",
        uncertainty=3e-7,
        note="measured in a Penning trap; the digits are read in Chin. Phys. Lett. 42, 063101 (2025)",
    ),
    _c(
        "D52.g_J",
        1.20036739,
        "",
        "ArnoldPRL2020",
        tag="verified",
        uncertainty=2.4e-8,
        note="measured on 138Ba+ as the Zeeman-splitting ratio 0.59943681(12) times g_S = 2.00249192(3)",
    ),
    _c(
        "D32.g_J_asd",
        0.79,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="cross-check only: the NIST ASD Lande column, 1.2% below the measured value and not a rounding of it",
    ),
    _c(
        "D52.g_J_asd",
        1.12,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="cross-check only, and wrong: 6.7% from the measured 1.20036739(24), the earlier 1.200372(4)(7) "
        "(Hoffman et al. 2013), the LS Lande 1.2005 and the isoelectronic 1.2003340 of Ca+",
    ),
    _c(
        "P32.g_J",
        1.32,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="cross-check only: no measured 6p 2P3/2 g_J exists; species() uses the Lande 4/3",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """The ``Species`` record for 137Ba+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t["ba137.D32.energy_cm"])
    e_d52 = energy_hz(t["ba137.D52.energy_cm"])
    e_p12 = energy_hz(t["ba137.P12.energy_cm"])
    e_p32 = energy_hz(t["ba137.P32.energy_cm"])
    s12 = Level(
        "S1/2",
        0.0,
        None,
        t["ba137.S12.A_hfs_hz"].value,
        0.0,
        t["ba137.S12.g_J"].value,
        ("BlattWerth1982", "Marx1998", "Stone2019"),
    )
    d32 = Level(
        "D3/2",
        e_d32,
        t["ba137.D32.lifetime_s"].value,
        t["ba137.D32.A_hfs_hz"].value,
        t["ba137.D32.B_hfs_hz"].value,
        t["ba137.D32.g_J"].value,
        ("NIST_ASD_5_12", "Lewty2013", "Knoll1996", "Yu1997"),
    )
    d52 = Level(
        "D5/2",
        e_d52,
        t["ba137.D52.lifetime_s"].value,
        t["ba137.D52.A_hfs_hz"].value,
        t["ba137.D52.B_hfs_hz"].value,
        t["ba137.D52.g_J"].value,
        ("NIST_ASD_5_12", "Lewty2013", "ArnoldPRL2020", "ZhangBa2020"),
    )
    p12 = Level(
        "P1/2",
        e_p12,
        t["ba137.P12.lifetime_s"].value,
        t["ba137.P12.A_hfs_hz"].value,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "Villemoes1993", "Arnold2019", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        t["ba137.P32.lifetime_s"].value,
        t["ba137.P32.A_hfs_hz"].value,
        t["ba137.P32.B_hfs_hz"].value,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Villemoes1993", "ZhangBa2020", "PLAN_background"),
    )

    g_p12 = gamma_hz_from_lifetime(t["ba137.P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t["ba137.P32.lifetime_s"])
    b12_d32 = t["ba137.P12.branching_to_D32"].value
    b32_d52 = t["ba137.P32.branching_to_D52"].value
    b32_d32 = t["ba137.P32.branching_to_D32"].value
    c12 = ("NIST_ASD_5_12", "Arnold2019")
    c32 = ("NIST_ASD_5_12", "ZhangBa2020")
    transitions = (
        Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), g_p12, 1.0 - b12_d32, "E1", c12),
        Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), g_p12, b12_d32, "E1", c12),
        Transition("S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), g_p32, 1.0 - b32_d52 - b32_d32, "E1", c32),
        Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b32_d52, "E1", c32),
        Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b32_d32, "E1", c32),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["ba137.mass_atomic_u"]),
        nuclear_spin=float(Fraction(t["ba137.nuclear_spin"].value).limit_denominator(2)),
        mu_I_nuclear_magnetons=t["ba137.mu_I_nuclear_magnetons"].value,
        levels=(s12, d32, d52, p12, p32),
        transitions=transitions,
        # this package's choice (the plan names none): the mF = 0 <-> mF = 0 ground-state pair, lower state first, with
        # the Ba+ scheme of Section 8.1 (493.5 nm cycling, 649.9 and 614.3 nm repumps, shelving through P3/2)
        qubit=("S1/2 F=1 mF=0", "S1/2 F=2 mF=0"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-P3/2",
    )
