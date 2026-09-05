"""40Ca+ species table (PLAN.md Sections 4.5.2, 4.5.7, 8.1, 9.13, 9.14, 9.15, 12; Appendix E).

I = 0, so there is no hyperfine structure and the Lande formula supplies g_J ([background]; the measured
g_S ~ 2.00226 and g_D ~ 1.20033 the plan mentions were "not verified here", Section 4.5.7). The metastable
D lifetimes are Kreuter et al. 2005's single-ion measurements with the theoretical values stored
separately as cross-checks and never as lifetimes (Section 4.5.7); the P-level linewidths are the
"quoted" 21.57 and 23.4 MHz of Section 9.13, whose total-versus-partial reading the plan leaves open, so
they are tagged ``contested`` and read here as TOTAL rates (the Appendix E field semantics), with the
Section 9.13 I_sat anchor 45.11 mW/cm^2 reproducible only under the PARTIAL reading (see the tests).
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited, Tag
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    MissingConstant,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    lifetime_s_from_linewidth,
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
        ledger_id=_P + suffix,
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
        note="'the quoted 21.57 MHz' of the 397 nm line, source not named in PLAN.md; whether it is the total P1/2 "
        "decay rate or the partial rate into S1/2 is unresolved ('the species table must say which', Section 9.13). "
        "THIS TABLE READS IT AS THE TOTAL RATE (Transition.gamma_hz semantics); Section 9.13's 2.045 e a0 and "
        "45.11 mW/cm^2 assume the partial reading",
    ),
    _c(
        "P32.linewidth_quoted_hz",
        23.4e6,
        "Hz",
        "PLAN_9_13",
        tag="contested",
        note="'the quoted 23.4 MHz' of the 393 nm line, same caveat; read here as the total P3/2 rate",
    ),
    _c(
        "P12.branching_to_D32",
        0.06,
        "",
        "PLAN_8_1",
        tag="contested",
        note="'the 6% branch to D3/2' (Section 8.1); Section 12 records 1:12, 1:16 and about 1:17.6 in different "
        "sources, a 30% uncertainty with no dedicated measurement verified in any run",
    ),
    _c(
        "P32.branching_to_D",
        1.0 / 17.0,
        "",
        "Ozeri2007",
        tag="contested",
        note="Ozeri Table I f^-1 = 17 for Ca+ (total P -> D branching); Section 9.13 uses 0.941 for the S1/2 branch. "
        "The split between D5/2 (854 nm) and D3/2 (850 nm) is not in PLAN.md",
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

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant(
        "measured P1/2 and P3/2 lifetimes (only 'quoted' linewidths with an unresolved total/partial reading)",
        "PLAN.md Section 9.13 names no source; a modern lifetime measurement must be cited",
    ),
    MissingConstant("P3/2 branching split between D5/2 (854 nm) and D3/2 (850 nm)", "not in PLAN.md"),
    MissingConstant(
        "Kreuter 2005 one-sided systematic corrections to the D lifetimes (Table III)", "Kreuter et al. 2005"
    ),
    MissingConstant(
        "measured g_J of S1/2 (~2.00226) and D5/2 (~1.20033)", "PLAN.md 4.5.7 mentions them as unverified"
    ),
)


def species() -> Species:
    """The Appendix E ``Species`` record for 40Ca+, built from :data:`TABLE` alone."""
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
        lifetime_s_from_linewidth(t[_P + "P12.linewidth_quoted_hz"]),
        0.0,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "PLAN_9_13", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        lifetime_s_from_linewidth(t[_P + "P32.linewidth_quoted_hz"]),
        0.0,
        0.0,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "PLAN_9_13", "PLAN_background"),
    )

    g_p12 = t[_P + "P12.linewidth_quoted_hz"].value
    g_p32 = t[_P + "P32.linewidth_quoted_hz"].value
    b_p12_d = t[_P + "P12.branching_to_D32"].value
    b_p32_d = t[_P + "P32.branching_to_D"].value
    s_p12 = Transition(
        "S1/2",
        "P1/2",
        wavelength_vac_m(0.0, e_p12),
        g_p12,
        1.0 - b_p12_d,
        "E1",
        ("NIST_ASD_5_12", "PLAN_9_13", "PLAN_8_1"),
    )
    d32_p12 = Transition(
        "D3/2",
        "P1/2",
        wavelength_vac_m(e_d32, e_p12),
        g_p12,
        b_p12_d,
        "E1",
        ("NIST_ASD_5_12", "PLAN_9_13", "PLAN_8_1"),
    )
    s_p32 = Transition(
        "S1/2",
        "P3/2",
        wavelength_vac_m(0.0, e_p32),
        g_p32,
        1.0 - b_p32_d,
        "E1",
        ("NIST_ASD_5_12", "PLAN_9_13", "Ozeri2007"),
    )
    s_d52 = Transition(
        "S1/2",
        "D5/2",
        wavelength_vac_m(0.0, e_d52),
        gamma_hz_from_lifetime(t[_P + "D52.lifetime_s"]),
        1.0,
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
        transitions=(s_p12, d32_p12, s_p32, s_d52, s_d32),
        qubit=("S1/2 mJ=-1/2", "D5/2 mJ=-1/2"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-D5/2",
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
