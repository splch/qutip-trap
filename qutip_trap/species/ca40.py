"""40Ca+ species table (PLAN.md Section 4.5.6).

I = 0, so there is no hyperfine structure; the g_J are Lande values ([background]; the measured S1/2 and D5/2 values,
both taken on 40Ca+, are in the 43Ca+ table). The metastable D lifetimes are Kreuter et al. 2005's single-ion
measurements (their one-sided systematic corrections are not transcribed), with the theory values stored beside them.

The P-level TOTAL rates come from measured lifetimes (Hettrich et al. 2015's 6.904(26) ns, Meir et al. 2020's
6.639(42) ns), and PLAN.md 9.13's "quoted" 21.57 and 23.4 MHz are read as PARTIAL rates into S1/2, as Hettrich prints
the first (gamma_PS = 2 pi x 21.57(8) MHz). With Ramm et al. 2013's branching the 397 nm partial rate is 21.5691 MHz,
reproducing Hettrich to 4e-5, so the plan's 2.045 e a0 and 45.11 mW/cm^2 come out of this table. The 393 nm pair does
not: Meir's lifetime with Gerritsma's branching gives 22.4071 MHz, 4.2% below the plan's unsourced 23.4 MHz.
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

NAME = "40Ca+"
_P = "ca40."
_c = CitedFactory(_P)

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 39.962590863, "u", "NIST_AWIC", uncertainty=2.2e-8),
    _c("nuclear_spin", 0.0, "hbar", "PLAN_background", tag="background", note="even-even nucleus"),
    _c(
        "D32.energy_cm", 13650.19, "cm^-1", "NIST_ASD_5_12", note="1e8/7325.91 A, Kreuter's vacuum wavelength"
    ),
    _c("D52.energy_cm", 13710.88, "cm^-1", "NIST_ASD_5_12", note="1e8/7293.48 A: the 729.347 nm line"),
    _c("P12.energy_cm", 25191.51, "cm^-1", "NIST_ASD_5_12", note="396.959 nm vacuum"),
    _c("P32.energy_cm", 25414.40, "cm^-1", "NIST_ASD_5_12", note="393.478 nm vacuum"),
    _c(
        "D32.lifetime_s",
        1.176,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="single-ion measurement, statistical error; the one-sided systematic corrections (repumping, "
        "detection error, heating) are not combined by the source",
    ),
    _c(
        "D52.lifetime_s",
        1.168,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.009,
        note="single-ion measurement, concordant with Barton et al. 2000's 1.168(7) s",
    ),
    _c("D52.lifetime_barton_s", 1.168, "s", "Barton2000", tag="verified", uncertainty=0.007),
    _c(
        "D32.lifetime_theory_s",
        1.196,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="all-order theory; a cross-check on the reduced element, never a lifetime",
    ),
    _c("D52.lifetime_theory_s", 1.165, "s", "Kreuter2005", tag="verified", uncertainty=0.011, note="theory"),
    _c(
        "S12_D32.quadrupole_element_au",
        7.939,
        "e a0^2",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.037,
        note="<v||Q||w> in the Johnson (1/15)(omega/c)^5 normalization with the UPPER-state degeneracy (the 1/75 "
        "convention's elements are sqrt(5) larger); reproduces 1195.7 ms",
    ),
    _c(
        "S12_D52.quadrupole_element_au",
        9.740,
        "e a0^2",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.047,
        note="Johnson normalization, upper-state degeneracy 6; reproduces 1165.5 ms",
    ),
    _c(
        "P12.linewidth_quoted_hz",
        21.57e6,
        "Hz",
        "PLAN_9_13",
        tag="contested",
        note="cross-check only: PLAN.md 9.13's unsourced 'quoted 21.57 MHz', read as the partial rate into S1/2 "
        "(as a total it would mean tau = 7.379 ns, which no measurement supports)",
    ),
    _c(
        "P32.linewidth_quoted_hz",
        23.4e6,
        "Hz",
        "PLAN_9_13",
        tag="contested",
        note="cross-check only: PLAN.md 9.13's unsourced 'quoted 23.4 MHz', which matches no measurement as a total "
        "or a partial rate (Meir 2020: 23.9727 / 22.4071 MHz; Jin and Church 1993: 22.9860 / 21.4848 MHz)",
    ),
    _c(
        "P12.branching_to_D32",
        0.06435,
        "",
        "Ramm2013",
        tag="verified",
        uncertainty=0.00007,
        note="0.93565(7) into S1/2, corroborated by Hettrich et al. 2015's 0.93572(25)",
    ),
    _c(
        "P12.branching_to_D32_plan_8_1",
        0.06,
        "",
        "PLAN_8_1",
        tag="contested",
        note="cross-check only: Section 8.1's rounder '6% branch to D3/2', 6.8% low",
    ),
    _c(
        "P32.branching_to_D_ozeri",
        1.0 / 17.0,
        "",
        "Ozeri2007",
        tag="contested",
        note="cross-check only: Ozeri Table I's f^-1 = 17 (total P -> D) is listed for 43Ca+; Gerritsma et al. "
        "2008 measure 0.06531",
    ),
    _c(
        "P32.branching_to_D52",
        0.0587,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.0002,
        note="854 nm, the D5/2 repump; the printed 0.9347(3)/0.0587(2)/0.00661(4) sum to 1.00001, so the S1/2 share "
        "is taken as the remainder 0.93469",
    ),
    _c(
        "P32.branching_to_D32",
        0.00661,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.00004,
        note="850 nm",
    ),
    _c(
        "P12.partial_rate_to_S12_hz",
        21.57e6,
        "Hz",
        "Hettrich2015",
        tag="verified",
        uncertainty=0.08e6,
        note="cross-check: the table reproduces it as 23.0526 x 0.93565 = 21.5691 MHz; Hettrich's "
        "<S1/2||d||P1/2> = 2.8928(43) e a0 is sqrt(2) x 2.0455, the normalization without the (2J+1)",
    ),
    _c(
        "P12.lifetime_s",
        6.904e-9,
        "s",
        "Hettrich2015",
        tag="verified",
        uncertainty=0.026e-9,
        note="gamma/2pi = 23.0526 MHz total; Jin and Church 1993's 7.098(20) ns is 7 sigma above it",
    ),
    _c(
        "P32.lifetime_s",
        6.639e-9,
        "s",
        "Meir2020",
        tag="verified",
        uncertainty=0.042e-9,
        note="gamma/2pi = 23.9727 MHz total, 22.4071 MHz partial into S1/2; Jin and Church 1993's 6.924(19) ns is "
        "6 sigma above it",
    ),
    _c(
        "D52_D32.M1_rate_s",
        2.45e-6,
        "s^-1",
        "AliKim1988_via_Kreuter2005",
        note="A12 of the D5/2 -> D3/2 magnetic-dipole line, a calculation quoted by Kreuter et al. 2005 Eq. 1 for the "
        "blackbody mixing rate W12 = A12 n_bar(nu, T); the level energies put the line at 1.8194 THz",
    ),
    _c(
        "D.collision_quench_H2_cm3_s",
        37e-12,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="quenching of the D levels by H2: a coefficient, a rate only as Gamma_s p_s/(k_B T)",
    ),
    _c("D.collision_quench_N2_cm3_s", 170e-12, "cm^3/s", "Knoop1995_via_Kreuter2005"),
    _c(
        "D.collision_j_mix_H2_cm3_s",
        3e-10,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="fine-structure (j-) mixing D5/2 <-> D3/2 by H2",
    ),
    _c("D.collision_j_mix_N2_cm3_s", 13e-10, "cm^3/s", "Knoop1995_via_Kreuter2005"),
    _c(
        "S12_D52.wavelength_vac_m",
        729.347e-9,
        "m",
        "James1998",
        tag="corrected",
        note="vacuum; James's Fig. 4 prints the air value 729.147 nm",
    ),
    _c("S12_D32.wavelength_vac_m", 732.591e-9, "m", "Kreuter2005", tag="verified", note="vacuum"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """The ``Species`` record for 40Ca+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t[_P + "D32.energy_cm"])
    e_d52 = energy_hz(t[_P + "D52.energy_cm"])
    e_p12 = energy_hz(t[_P + "P12.energy_cm"])
    e_p32 = energy_hz(t[_P + "P32.energy_cm"])
    lande = ("NIST_ASD_5_12", "PLAN_background")

    s12 = Level("S1/2", 0.0, None, 0.0, 0.0, lande_g_j(0, half, half), lande)
    d32 = Level(
        "D3/2",
        e_d32,
        t[_P + "D32.lifetime_s"].value,
        0.0,
        0.0,
        lande_g_j(2, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Kreuter2005", "PLAN_background"),
    )
    d52 = Level(
        "D5/2",
        e_d52,
        t[_P + "D52.lifetime_s"].value,
        0.0,
        0.0,
        lande_g_j(2, half, Fraction(5, 2)),
        ("NIST_ASD_5_12", "Kreuter2005", "Barton2000", "PLAN_background"),
    )
    p12 = Level(
        "P1/2",
        e_p12,
        t[_P + "P12.lifetime_s"].value,
        0.0,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "Hettrich2015", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        t[_P + "P32.lifetime_s"].value,
        0.0,
        0.0,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Meir2020", "PLAN_background"),
    )

    # the TOTAL rates come from the measured lifetimes, never from the plan's quoted (partial) linewidths
    g_p12 = gamma_hz_from_lifetime(t[_P + "P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t[_P + "P32.lifetime_s"])
    b_p12_d = t[_P + "P12.branching_to_D32"].value
    b_p32_d52 = t[_P + "P32.branching_to_D52"].value
    b_p32_d32 = t[_P + "P32.branching_to_D32"].value
    p12_cites = ("NIST_ASD_5_12", "Hettrich2015", "Ramm2013")
    s_p12 = Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), g_p12, 1.0 - b_p12_d, "E1", p12_cites)
    d32_p12 = Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), g_p12, b_p12_d, "E1", p12_cites)
    p32_cites = ("NIST_ASD_5_12", "Meir2020", "Gerritsma2008")
    s_p32 = Transition(
        "S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), g_p32, 1.0 - b_p32_d52 - b_p32_d32, "E1", p32_cites
    )
    d52_p32 = Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b_p32_d52, "E1", p32_cites)
    d32_p32 = Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b_p32_d32, "E1", p32_cites)
    # the D5/2 -> D3/2 M1 branch carries A12 tau(D5/2) = 2.9e-6 of the decay (the line the blackbody mixing of
    # MetastableChannels multiplies by the Bose occupation); the E2 branch into S1/2 carries the rest
    b_m1 = t[_P + "D52_D32.M1_rate_s"].value * t[_P + "D52.lifetime_s"].value
    d32_d52 = Transition(
        "D3/2",
        "D5/2",
        wavelength_vac_m(e_d32, e_d52),
        gamma_hz_from_lifetime(t[_P + "D52.lifetime_s"]),
        b_m1,
        "M1",
        ("AliKim1988_via_Kreuter2005", "Kreuter2005", "NIST_ASD_5_12"),
    )
    s_d52 = Transition(
        "S1/2",
        "D5/2",
        wavelength_vac_m(0.0, e_d52),
        gamma_hz_from_lifetime(t[_P + "D52.lifetime_s"]),
        1.0 - b_m1,
        "E2",
        ("NIST_ASD_5_12", "Kreuter2005", "Barton2000", "James1998"),
    )
    s_d32 = Transition(
        "S1/2",
        "D3/2",
        wavelength_vac_m(0.0, e_d32),
        gamma_hz_from_lifetime(t[_P + "D32.lifetime_s"]),
        1.0,
        "E2",
        ("NIST_ASD_5_12", "Kreuter2005"),
    )

    return Species(
        name=NAME,
        mass_u=ion_mass_u(t[_P + "mass_atomic_u"]),
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=(s12, d32, d52, p12, p32),
        transitions=(s_p12, d32_p12, s_p32, d52_p32, d32_p32, s_d52, s_d32, d32_d52),
        qubit=("S1/2 mJ=-1/2", "D5/2 mJ=-1/2"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-D5/2",
    )
