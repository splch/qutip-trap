"""88Sr+ species table (PLAN.md Section 4.5.6).

I = 0 (Lande g_J, [background]). The 674 nm quadrupole line takes its vacuum wavelength from the clock frequency (1.3 ppb),
not the 0.01 cm^-1 level energy (5.4e-7 away), with Letchumanan et al. 2005's D5/2 lifetime and an LS-estimated 9e-5 M1
branch to D3/2. The P lifetimes and branchings are Pinnington et al. 1995's and Zhang et al. 2016's. Not tabulated: the
D3/2 lifetime (about 435 ms, Mannervik et al. 1999).
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
from qutip_trap.units import C_M_PER_S, lande_g_j

NAME = "88Sr+"
_c = CitedFactory("sr88.")

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
        note="674.0252 nm vacuum, 5.4e-7 from the clock-frequency wavelength the 674 nm transition uses",
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
        note="160,000 single-ion shelving periods, corrections 0.11% of the rate (the thesis prints an uncorrected "
        "390.3 ms); Gerz 1987's superseded 345(33) ms is the negative control",
    ),
    _c(
        "D52.A_per_s",
        2.559,
        "s^-1",
        "Sansonetti2012",
        tag="verified",
        uncertainty=0.010,
        note="NIST Sr II compilation; equals 1/0.3908 s (its accuracy grade not verified)",
    ),
    _c(
        "clock_frequency_hz",
        444_779_044_095_484.6,
        "Hz",
        "Sansonetti2012",
        tag="verified",
        note="the S1/2 - D5/2 clock frequency: lambda_vac = 674.025591 nm, where NIST's air Ritz 673.8392 nm is 276 "
        "ppm off",
    ),
    _c(
        "D52.M1_branch_to_D32",
        9e-5,
        "",
        "PLAN_background",
        tag="background",
        note="LS-coupling estimate of the D5/2 -> D3/2 M1 channel (2.4e-4 s^-1), no source",
    ),
    _c(
        "P12.lifetime_s",
        7.39e-9,
        "s",
        "Pinnington1995",
        tag="extracted",
        uncertainty=0.07e-9,
        note="gamma/2pi = 21.539 MHz; quoted verbatim by Likforman et al. 2016, the NIST ASD A values give 7.388 ns",
    ),
    _c(
        "P32.lifetime_s",
        6.63e-9,
        "s",
        "Pinnington1995",
        tag="extracted",
        uncertainty=0.07e-9,
        note="gamma/2pi = 24.008 MHz; quoted verbatim by Zhang et al. 2016",
    ),
    _c(
        "P12.branching_to_D32",
        0.05502,
        "",
        "Zhang2016",
        tag="verified",
        uncertainty=0.00008,
        note="1091.8 nm, the D3/2 repump; Likforman et al. 2016's 0.9449(5) into S1/2 agrees",
    ),
    _c("P32.branching_to_D32", 0.0063, "", "Zhang2016", tag="verified", uncertainty=0.0003, note="1004 nm"),
    _c(
        "P32.branching_to_D52",
        0.0531,
        "",
        "Zhang2016",
        tag="verified",
        uncertainty=0.0002,
        note="1033 nm, the D5/2 repump and the only blackbody deshelving channel; the printed 0.9406/0.0063/0.0531 "
        "sum to 1",
    ),
    _c(
        "D52_P32.A_gallagher_per_s",
        8.7e6,
        "s^-1",
        "Sansonetti2012",
        tag="contested",
        uncertainty=1.5e6,
        note="A(1033 nm) of NIST JPCRD Table 2 (Gallagher 1967, accuracy C+); Zhang et al. 2016's 8.010(89)e6 s^-1 "
        "is 8% lower; a cross-check",
    ),
    _c(
        "D52.lifetime_theory_jiang_s",
        0.394,
        "s",
        "Jiang2009",
        tag="verified",
        uncertainty=0.003,
        note="theory; a bracket",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


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
    # the 674 nm wavelength from the clock frequency; Species.__post_init__ cross-checks it with the level energies at 1e-6
    s_d52 = Transition(
        "S1/2",
        "D5/2",
        C_M_PER_S / t["sr88.clock_frequency_hz"].value,
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
