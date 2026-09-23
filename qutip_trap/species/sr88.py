"""88Sr+ species table (I = 0: Lande g_J).

The 674 nm S1/2-D5/2 wavelength comes from the stored clock frequency, not the level energies; the D5/2 -> D3/2 M1
branch is an LS-coupling estimate, and the D3/2 lifetime is not tabulated.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    Cited,
    MissingConstant,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)
from qutip_trap.units import C_M_PER_S, lande_g_j

NAME = "88Sr+"
_c = cited_factory("sr88.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 87.9056125, "u", "NIST_AWIC", uncertainty=1.2e-6),
    _c("nuclear_spin", 0.0, "hbar", "PLAN_background", tag="background", note="even-even nucleus"),
    _c("D32.energy_cm", 14555.90, "cm^-1", "NIST_ASD_5_12", uncertainty=0.01),
    _c(
        "D52.energy_cm",
        14836.24,
        "cm^-1",
        "NIST_ASD_5_12",
        uncertainty=0.01,
        note="674.0252 nm vacuum from this level energy alone, which is 5.4e-7 relative from the "
        "674.025591 nm the stored clock frequency gives (an earlier note here claimed 1e-7, a 5x "
        "overstatement); species() now takes the CLOCK FREQUENCY for the 674 nm wavelength, this energy "
        "entering only the level record (audit item E13b)",
    ),
    _c("P12.energy_cm", 23715.19, "cm^-1", "NIST_ASD_5_12", uncertainty=0.01, note="421.67 nm vacuum"),
    _c("P32.energy_cm", 24516.65, "cm^-1", "NIST_ASD_5_12", uncertainty=0.01, note="407.89 nm vacuum"),
    _c(
        "D52.lifetime_s",
        0.3908,
        "s",
        "Letchumanan2005",
        tag="verified",
        uncertainty=0.0016,
        note="160,000 single-ion shelving periods; corrections total 0.11% of the rate; the thesis Table 5.2 prints an "
        "uncorrected 390.3 ms (PLAN.md 4.5.7); the superseded 345(33) ms of Gerz 1987 is the negative control",
    ),
    _c(
        "D52.A_per_s",
        2.559,
        "s^-1",
        "Sansonetti2012",
        tag="verified",
        uncertainty=0.010,
        note="NIST Sr II compilation (Sansonetti 2012); equals 1/0.3908 s. An earlier note here claimed "
        "'accuracy AA'; the accuracy grade of this E2 entry was not verified and the claim is withdrawn -- the "
        "uncertainty carried is the one implied by the D5/2 lifetime",
    ),
    _c(
        "clock_frequency_hz",
        444_779_044_095_484.6,
        "Hz",
        "Sansonetti2012",
        tag="verified",
        note="the 88Sr II S1/2 - D5/2 clock frequency, giving lambda_vac = 674.025591 nm; NIST's air Ritz value "
        "673.8392 nm is the 276 ppm trap (PLAN.md 4.5.7)",
    ),
    _c(
        "D52.M1_branch_to_D32",
        9e-5,
        "",
        "PLAN_background",
        tag="background",
        note="LS-coupling estimate of the D5/2 -> D3/2 M1 channel (2.4e-4 s^-1), no source (PLAN.md 4.5.7)",
    ),
    _c(
        "P12.lifetime_s",
        7.39e-9,
        "s",
        "Pinnington1995",
        tag="extracted",
        uncertainty=0.07e-9,
        note="gamma/2pi = 21.539 MHz total. NOT primary-verified: IOP blocks automated retrieval, so the "
        "digits reach this table quoted verbatim by Likforman et al. 2016 and corroborated by the NIST ASD "
        "A values (1.279e8 + 7.46e6 = 1.3536e8 s^-1 gives 7.388 ns)",
    ),
    _c(
        "P32.lifetime_s",
        6.63e-9,
        "s",
        "Pinnington1995",
        tag="extracted",
        uncertainty=0.07e-9,
        note="gamma/2pi = 24.008 MHz total; same provenance caveat, quoted verbatim by Zhang et al. 2016",
    ),
    _c(
        "P12.branching_to_D32",
        0.05502,
        "",
        "Zhang2016",
        tag="verified",
        uncertainty=0.00008,
        note="1091.8 nm, the designated D3/2 repump; the S1/2 share is 0.94498(8) and the two sum to 1 "
        "exactly. Likforman et al. 2016's independent 0.9449(5) agrees",
    ),
    _c(
        "P32.branching_to_D32",
        0.0063,
        "",
        "Zhang2016",
        tag="verified",
        uncertainty=0.0003,
        note="1004 nm",
    ),
    _c(
        "P32.branching_to_D52",
        0.0531,
        "",
        "Zhang2016",
        tag="verified",
        uncertainty=0.0002,
        note="1033 nm, the designated D5/2 repump and the only blackbody deshelving channel (PLAN.md 4.5.7). "
        "The three printed fractions 0.9406(2)/0.0063(3)/0.0531(2) sum to 1.0000, so the S1/2 share is taken "
        "as 1 - 0.0063 - 0.0531 = 0.9406 exactly. NIST ASD's implied D5/2 fraction 0.0577 (from Gallagher "
        "1967's A = 8.7(15)e6 s^-1) is 8.7% higher; Zhang's A = 8.010(89)e6 s^-1 supersedes it 17x tighter",
    ),
    _c(
        "D52_P32.A_gallagher_per_s",
        8.7e6,
        "s^-1",
        "Sansonetti2012",
        tag="contested",
        uncertainty=1.5e6,
        note="the A(1033 nm) of NIST JPCRD Table 2 (accuracy C+, <=18%, reference 67GAL = Gallagher 1967) that "
        "validation/scripts/check_sr_lifetime_bbr.py uses for the blackbody deshelving rate; Zhang et al. "
        "2016's 8.010(89)e6 s^-1 is 8% lower and 17x tighter. Cross-check only",
    ),
    _c(
        "D52.lifetime_theory_jiang_s",
        0.394,
        "s",
        "Jiang2009",
        tag="verified",
        uncertainty=0.003,
        note="theory; bracket only",
    ),
)

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant(
        "D3/2 lifetime (about 435 ms)",
        "Mannervik et al., Phys. Rev. Lett. 83, 698 (1999), the storage-ring measurement; read the digits "
        "before entering",
    ),
    MissingConstant("measured g_J values (Lande used)", "not in PLAN.md"),
)


def species() -> Species:
    """The ``Species`` record for 88Sr+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t["sr88.D32.energy_cm"])
    e_d52 = energy_hz(t["sr88.D52.energy_cm"])
    e_p12 = energy_hz(t["sr88.P12.energy_cm"])
    e_p32 = energy_hz(t["sr88.P32.energy_cm"])
    lande = ("NIST_ASD_5_12", "PLAN_background")
    p_cites = ("NIST_ASD_5_12", "Pinnington1995", "Zhang2016", "PLAN_background")
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, lande_g_j(0, half, half), lande),
        Level("D3/2", e_d32, None, 0.0, 0.0, lande_g_j(2, half, Fraction(3, 2)), lande),
        Level(
            "D5/2",
            e_d52,
            t["sr88.D52.lifetime_s"].value,
            0.0,
            0.0,
            lande_g_j(2, half, Fraction(5, 2)),
            ("NIST_ASD_5_12", "Letchumanan2005", "PLAN_background"),
        ),
        Level("P1/2", e_p12, t["sr88.P12.lifetime_s"].value, 0.0, 0.0, lande_g_j(1, half, half), p_cites),
        Level(
            "P3/2",
            e_p32,
            t["sr88.P32.lifetime_s"].value,
            0.0,
            0.0,
            lande_g_j(1, half, Fraction(3, 2)),
            p_cites,
        ),
    )
    # the 674 nm wavelength comes from the clock frequency; the level energy is 5.4e-7 away (Species checks 1e-6)
    lam_clock = C_M_PER_S / t["sr88.clock_frequency_hz"].value
    s_d52 = Transition(
        "S1/2",
        "D5/2",
        lam_clock,
        gamma_hz_from_lifetime(t["sr88.D52.lifetime_s"]),
        1.0 - t["sr88.D52.M1_branch_to_D32"].value,
        "E2",
        ("NIST_ASD_5_12", "Letchumanan2005", "Sansonetti2012", "PLAN_background"),
    )
    g_p12 = gamma_hz_from_lifetime(t["sr88.P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t["sr88.P32.lifetime_s"])
    b12_d32 = t["sr88.P12.branching_to_D32"].value
    b32_d32 = t["sr88.P32.branching_to_D32"].value
    b32_d52 = t["sr88.P32.branching_to_D52"].value
    s_p12 = Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), g_p12, 1.0 - b12_d32, "E1", p_cites)
    d32_p12 = Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), g_p12, b12_d32, "E1", p_cites)
    s_p32 = Transition(
        "S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), g_p32, 1.0 - b32_d32 - b32_d52, "E1", p_cites
    )
    d32_p32 = Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b32_d32, "E1", p_cites)
    d52_p32 = Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b32_d52, "E1", p_cites)
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["sr88.mass_atomic_u"]),
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=levels,
        transitions=(s_d52, s_p12, d32_p12, s_p32, d32_p32, d52_p32),
        qubit=("S1/2 mJ=-1/2", "D5/2 mJ=-1/2"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-D5/2",
    )
