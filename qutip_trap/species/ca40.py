"""40Ca+ species table (I = 0: no hyperfine structure, Lande g_J).

The P-level TOTAL rates come from measured lifetimes (Hettrich et al. 2015, Meir et al. 2020); the quoted 21.57 and
23.4 MHz linewidths are read as partial rates into S1/2 and kept as cross-checks. The D lifetimes are Kreuter et al.
2005's measurements; theory values are stored as cross-checks, never as lifetimes.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    Cited,
    MissingConstant,
    Tag,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "40Ca+"
_P = "ca40."


def _c(
    suffix: str,
    value: float,
    unit: str,
    source: str,
    *,
    tag: Tag = "extracted",
    uncertainty: float | None = None,
    note: str = "",
) -> Cited:
    return Cited(
        value=value,
        unit=unit,
        source=source,
        key=_P + suffix,
        tag=tag,
        uncertainty=uncertainty,
        note=note,
    )


_ENTRIES: tuple[Cited, ...] = (
    _c(
        "mass_atomic_u",
        39.962590863,
        "u",
        "NIST_AWIC",
        uncertainty=2.2e-8,
        note="PLAN.md 9.15 uses 39.9626 u",
    ),
    _c(
        "nuclear_spin",
        0.0,
        "hbar",
        "PLAN_background",
        tag="background",
        note="even-even nucleus, no hyperfine structure",
    ),
    _c(
        "D32.energy_cm",
        13650.19,
        "cm^-1",
        "NIST_ASD_5_12",
        note="1e8/7325.91 A, Kreuter's NIST vacuum wavelength",
    ),
    _c(
        "D52.energy_cm",
        13710.88,
        "cm^-1",
        "NIST_ASD_5_12",
        note="1e8/7293.48 A; the 729.347 nm of PLAN.md 4.5.7",
    ),
    _c(
        "P12.energy_cm",
        25191.51,
        "cm^-1",
        "NIST_ASD_5_12",
        note="396.959 nm vacuum; the plan's '397 nm' line",
    ),
    _c(
        "P32.energy_cm",
        25414.40,
        "cm^-1",
        "NIST_ASD_5_12",
        note="393.478 nm vacuum; the plan's '393 nm' line",
    ),
    _c(
        "D32.lifetime_s",
        1.176,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="single-ion measurement, statistical error; the one-sided systematic corrections of a few ms "
        "(repumping, detection error, motional heating) are not quadrature-combined by the source and are "
        "not transcribed in PLAN.md (see MISSING)",
    ),
    _c(
        "D52.lifetime_s",
        1.168,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.009,
        note="single-ion measurement; concordant with Barton et al. 2000's 1.168(7) s; PLAN.md 4.5.7 keeps 1.168 s "
        "as tau_upper of the 729 nm quadrupole record",
    ),
    _c(
        "D52.lifetime_barton_s",
        1.168,
        "s",
        "Barton2000",
        tag="verified",
        uncertainty=0.007,
        note="the concordant second measurement",
    ),
    _c(
        "D32.lifetime_theory_s",
        1.196,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="all-order theory; a cross-check on the Racah-normalized reduced element, never a lifetime",
    ),
    _c(
        "D52.lifetime_theory_s",
        1.165,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="all-order theory; cross-check only",
    ),
    _c(
        "S12_D32.quadrupole_element_au",
        7.939,
        "e a0^2",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.037,
        note="reduced quadrupole element <v||Q||w> in the Johnson (1/15)(omega/c)^5 normalization with the UPPER-state "
        "degeneracy; the (1/75) convention's elements are sqrt(5) larger (Section 13); reproduces 1195.7 ms",
    ),
    _c(
        "S12_D52.quadrupole_element_au",
        9.740,
        "e a0^2",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.047,
        note="Johnson normalization, upper-state degeneracy 6; reproduces 1165.5 ms (check_constants.py)",
    ),
    _c(
        "P12.linewidth_quoted_hz",
        21.57e6,
        "Hz",
        "PLAN_9_13",
        tag="contested",
        note="CROSS-CHECK ONLY (nothing is built from it): 'the quoted 21.57 MHz' of the 397 nm line, for "
        "which PLAN.md 9.13 names no source. THIS TABLE READS IT AS THE PARTIAL RATE into S1/2, because "
        "(a) Hettrich et al. 2015 print exactly gamma_PS = 2 pi x 21.57(8) MHz as that partial rate, "
        "(b) Section 9.13 itself reads it that way -- its 2.045 e a0 and 45.11 mW/cm^2 are the partial-rate "
        "closed forms -- and (c) as a TOTAL rate it would mean tau(P1/2) = 7.379 ns, contradicted by every "
        "lifetime measurement (Hettrich 6.904(26) ns, Jin and Church 1993 7.098(20) ns). The value the table "
        "USES is P12.partial_rate_to_S12_hz, reproduced from P12.lifetime_s x the S1/2 branching; the total "
        "is 23.0526 MHz, not 21.57. Ledger conv.ca40_linewidth_reading",
    ),
    _c(
        "P32.linewidth_quoted_hz",
        23.4e6,
        "Hz",
        "PLAN_9_13",
        tag="contested",
        note="CROSS-CHECK ONLY: 'the quoted 23.4 MHz' of the 393 nm line, source not named in PLAN.md, and it "
        "matches NO measurement under either reading -- Meir et al. 2020's tau = 6.639(42) ns gives 23.9727 MHz "
        "total and 22.4071 MHz partial, Jin and Church 1993's 6.924(19) ns gives 22.9860 / 21.4848. PLAN.md "
        "9.13 reads it as a partial rate (its 2.972 e a0 / 50.25 mW/cm^2), so this table reads it the same way "
        "for consistency with the 397 nm line, but takes the P3/2 total from Meir's lifetime, which means the "
        "plan's 393 nm anchors are NOT reproduced from the table (2.9085 e a0 / 48.113 mW/cm^2 at the plan's "
        "own air wavelength). Ledger anchor.ca40.p32_linewidth_readings",
    ),
    _c(
        "P12.branching_to_D32",
        0.06435,
        "",
        "Ramm2013",
        tag="verified",
        uncertainty=0.00007,
        note="the dedicated 40Ca+ P1/2 branching measurement (0.93565(7) into S1/2), corroborated by Hettrich "
        "et al. 2015's independent 0.93572(25). ADOPTED as the value the table uses in place of Section 8.1's "
        "rounder '6% branch to D3/2' (kept as P12.branching_to_D32_plan_8_1): with P12.lifetime_s it makes the "
        "partial rate into S1/2 come out at 21.5691 MHz, reproducing Hettrich's printed 21.57(8) MHz to 4e-5, "
        "which is what closes the total-versus-partial question of Section 9.13. Ledger conv.ca40_branching",
    ),
    _c(
        "P12.branching_to_D32_plan_8_1",
        0.06,
        "",
        "PLAN_8_1",
        tag="contested",
        note="CROSS-CHECK ONLY: 'the 6% branch to D3/2' (Section 8.1), the value Ramm et al. 2013's 0.06435(7) "
        "replaced; 6.8% low, and it moves the 397 nm partial rate by 0.5%. Section 12 records 1:12, 1:16 and "
        "about 1:17.6 in different sources, a 30% spread it calls a gap 'with no dedicated measurement' -- the "
        "dedicated measurement does exist and is now what the table uses",
    ),
    _c(
        "P32.branching_to_D_ozeri",
        1.0 / 17.0,
        "",
        "Ozeri2007",
        tag="contested",
        note="Ozeri Table I f^-1 = 17 (total P -> D branching), which PLAN.md:457 lists for 43Ca+ and NOT for "
        "40Ca+ -- the plan's 0.9412 S1/2 branch is that misattribution. SUPERSEDED as the value used by "
        "Gerritsma et al. 2008's dedicated measurement (total D branching 0.06531, 11% larger); kept as a "
        "contested cross-check only. It never entered the plan's Section 9.13 anchor 50.25 mW/cm^2, which is "
        "the closed form at a PARTIAL rate of 23.4 MHz and so carries no branching at all",
    ),
    _c(
        "P32.branching_to_D52",
        0.0587,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.0002,
        note="854 nm, the D5/2 repump that flips readout polarity; the dedicated measurement Section 12 "
        "records as missing. The three printed fractions 0.9347(3)/0.0587(2)/0.00661(4) sum to 1.00001, within "
        "their combined uncertainty, so the S1/2 share is taken as 1 - 0.0587 - 0.00661 = 0.93469 and "
        "Gerritsma's printed 0.9347 is reproduced to 1e-5",
    ),
    _c(
        "P32.branching_to_D32",
        0.00661,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.00004,
        note="850 nm; the fourth digit matters (0.0066 is 0.15% low)",
    ),
    _c(
        "P12.partial_rate_to_S12_hz",
        21.57e6,
        "Hz",
        "Hettrich2015",
        tag="verified",
        uncertainty=0.08e6,
        note="Hettrich et al. 2015's PARTIAL rate P1/2 -> S1/2, gamma_PS = 2 pi x 21.57(8) MHz (with "
        "gamma_PD = 2 pi x 1.482(8) MHz and tau = 6.904(26) ns, so the total is 23.05 MHz). This settles what "
        "PLAN.md 9.13's 'quoted 21.57 MHz' is, and the table now REPRODUCES it rather than storing it: "
        "gamma/2pi(P1/2) x b(P1/2 -> S1/2) = 23.0526 x 0.93565 = 21.5691 MHz, 4.0e-5 below the printed value "
        "and 0.02 sigma inside its 0.08 MHz uncertainty (asserted in tests/test_species_tables.py). Hettrich's "
        "printed <S1/2||d||P1/2> = 2.8928(43) e a0 is the same element in the normalization that omits the "
        "(2J+1) factor: sqrt(2) x 2.0455 = 2.8920, inside the printed uncertainty. Ledger "
        "conv.ca40_linewidth_reading",
    ),
    _c(
        "P12.lifetime_s",
        6.904e-9,
        "s",
        "Hettrich2015",
        tag="verified",
        uncertainty=0.026e-9,
        note="the modern single-ion P1/2 lifetime and the source of the TOTAL rate this table uses: "
        "gamma/2pi = 1/(2 pi tau) = 23.0526 MHz. Jin and Church 1993's 7.098(20) ns is 7 sigma above it and "
        "Hettrich states it disagrees with theory by more than 11 sigma; both bracket the plan's implied "
        "7.379 ns out of existence, which is why the quoted 21.57 MHz cannot be the total rate",
    ),
    _c(
        "P32.lifetime_s",
        6.639e-9,
        "s",
        "Meir2020",
        tag="verified",
        uncertainty=0.042e-9,
        note="the modern single-ion P3/2 lifetime and the source of the TOTAL rate this table uses: "
        "gamma/2pi = 23.9727 MHz, of which 0.93469 x 23.9727 = 22.4071 MHz is the partial rate into S1/2. Jin "
        "and Church 1993's 6.924(19) ns is 6 sigma above it (22.9860 total, 21.4848 partial). Neither matches "
        "the plan's unsourced 23.4 MHz under either reading (ledger anchor.ca40.p32_linewidth_readings)",
    ),
    _c(
        "D52_D32.M1_rate_s",
        2.45e-6,
        "s^-1",
        "AliKim1988_via_Kreuter2005",
        note="A12 of the 3d 2D5/2 -> 3d 2D3/2 magnetic-dipole line, a multiconfiguration Dirac-Fock calculation quoted by "
        "Kreuter et al. 2005 Eq. 1 for the blackbody mixing rate W12 = A12 n_bar(nu, T); the level energies put the line "
        "at 1.8194 THz where Kreuter quotes 1.82 THz (Section 4.5.7: W12 = 7.249e-6 s^-1 at 300.0 K with 1.82 THz)",
    ),
    _c(
        "D.collision_quench_H2_cm3_s",
        37e-12,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="collisional quenching of the metastable D levels by H2, a specific coefficient (cm^3 s^-1) that becomes a "
        "rate only as Gamma_s p_s/(k_B T) (Section 13 'Collision-induced rates')",
    ),
    _c(
        "D.collision_quench_N2_cm3_s",
        170e-12,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="collisional quenching by N2",
    ),
    _c(
        "D.collision_jmix_H2_cm3_s",
        3e-10,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="fine-structure (j-) mixing D5/2 <-> D3/2 by H2; the j-mixing coefficients run 7.73x the quenching ones "
        "(Section 9.16 'Collision-rate construction')",
    ),
    _c(
        "D.collision_jmix_N2_cm3_s",
        13e-10,
        "cm^3/s",
        "Knoop1995_via_Kreuter2005",
        note="fine-structure mixing by N2",
    ),
    _c(
        "S12_D52.wavelength_vac_m",
        729.347e-9,
        "m",
        "James1998",
        tag="corrected",
        note="vacuum; James's Fig. 4 prints the air value 729.147 nm (PLAN.md 4.5.7); cross-checked against the NIST levels",
    ),
    _c(
        "S12_D32.wavelength_vac_m",
        732.591e-9,
        "m",
        "Kreuter2005",
        tag="verified",
        note="NIST vacuum 7325.91 A",
    ),
)

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant(
        "a source for the 393 nm 'quoted 23.4 MHz' of PLAN.md 9.13",
        "no measurement matches it: Meir et al. 2020's tau(P3/2) = 6.639(42) ns gives 23.9727 MHz total and "
        "22.4071 MHz partial into S1/2, Jin and Church 1993's 6.924(19) ns gives 22.9860 / 21.4848. The table "
        "uses Meir's lifetime, so the plan's 393 nm anchors are not reproduced from it (ledger "
        "anchor.ca40.p32_linewidth_readings)",
    ),
    MissingConstant(
        "Kreuter 2005 one-sided systematic corrections to the D lifetimes (Table III)", "Kreuter et al. 2005"
    ),
    MissingConstant(
        "measured g_J of S1/2 (~2.00226) and D5/2 (~1.20033)", "PLAN.md 4.5.7 mentions them as unverified"
    ),
)


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

    # TOTAL rates from the measured lifetimes, never from the quoted linewidths (partial rates into S1/2)
    g_p12 = gamma_hz_from_lifetime(t[_P + "P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t[_P + "P32.lifetime_s"])
    b_p12_d = t[_P + "P12.branching_to_D32"].value
    b_p32_d52 = t[_P + "P32.branching_to_D52"].value
    b_p32_d32 = t[_P + "P32.branching_to_D32"].value
    p12_cites = ("NIST_ASD_5_12", "Hettrich2015", "Ramm2013")
    s_p12 = Transition(
        "S1/2",
        "P1/2",
        wavelength_vac_m(0.0, e_p12),
        g_p12,
        1.0 - b_p12_d,
        "E1",
        p12_cites,
    )
    d32_p12 = Transition(
        "D3/2",
        "P1/2",
        wavelength_vac_m(e_d32, e_p12),
        g_p12,
        b_p12_d,
        "E1",
        p12_cites,
    )
    p32_cites = ("NIST_ASD_5_12", "Meir2020", "Gerritsma2008")
    s_p32 = Transition(
        "S1/2",
        "P3/2",
        wavelength_vac_m(0.0, e_p32),
        g_p32,
        1.0 - b_p32_d52 - b_p32_d32,
        "E1",
        p32_cites,
    )
    # the 854 nm D5/2 repump line and the 850 nm D3/2 branch (Gerritsma et al. 2008)
    d52_p32 = Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b_p32_d52, "E1", p32_cites)
    d32_p32 = Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b_p32_d32, "E1", p32_cites)
    # the D5/2 -> D3/2 M1 branch carries A12 tau(D5/2) = 2.9e-6 of the decay; the E2 branch into S1/2 the rest
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
        quadrupole_element_au=t[_P + "S12_D52.quadrupole_element_au"].value,
        quadrupole_convention="johnson_1_15",
    )
    s_d32 = Transition(
        "S1/2",
        "D3/2",
        wavelength_vac_m(0.0, e_d32),
        gamma_hz_from_lifetime(t[_P + "D32.lifetime_s"]),
        1.0,
        "E2",
        ("NIST_ASD_5_12", "Kreuter2005"),
        quadrupole_element_au=t[_P + "S12_D32.quadrupole_element_au"].value,
        quadrupole_convention="johnson_1_15",
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
