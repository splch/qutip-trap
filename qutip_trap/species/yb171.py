"""171Yb+ species table (PLAN.md Section 4.5.6).

Level energies are NIST ASD 5.12 values in cm^-1; the hyperfine constants, lifetimes and branchings are the primary
measurements. The ground-state g_J is Han et al. 2025's calculation, and the 6p 2P levels carry the Lande values
([background]): NIST ASD's 0.667 and 1.333 for them are 2/3 and 4/3, the Lande factors at g_S = 2, which Meggers 1967's
Zeeman observations match to their three digits. The 5d 2D, 4f13 6s2 2F7/2 and 4f13 5d6s bracket levels keep Meggers's
observed g as NIST ASD lists it, a measurement good to a few 1e-3 (his ground state's 1.998 is 4.6e-3 below the calculated
2.002615) that departs from the Lande values in the third digit. Not tabulated: the D5/2 and F7/2 decay branchings
(Feldker et al. 2018, Tan et al. 2021) and the 1[5/2]5/2 lifetime.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    CitedFactory,
    a_hfs_from_two_manifold_splitting,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "171Yb+"
_P = "yb171."
_c = CitedFactory(_P)

_ENTRIES: tuple[Cited, ...] = (
    _c(
        "mass_atomic_u",
        170.9363302,
        "u",
        "NIST_AWIC",
        uncertainty=2.2e-6,
        note="the neutral atom; the ion mass is 170.93578 u",
    ),
    _c("nuclear_spin", 0.5, "hbar", "Olmschenk2007", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        0.49367,
        "mu_N",
        "PLAN_4_5_1",
        note="the value PLAN.md 4.5.1 uses for g_I; the primary measurement is not named there",
    ),
    _c(
        "S12.hfs_splitting_hz",
        12_642_812_118.5,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        note="the ZERO-FIELD splitting (= A for I = J = 1/2); the 12.642815 to 12.642821 GHz printed elsewhere are "
        "second-order-shifted values at 3 to 5 G",
    ),
    _c(
        "S12.g_J",
        2.002615,
        "",
        "Han2025",
        tag="corrected",
        uncertainty=7e-5,
        note="a calculation; no modern measurement exists (NIST ASD lists the spectroscopic 1.998)",
    ),
    _c("P12.energy_cm", 27061.82, "cm^-1", "NIST_ASD_5_12"),
    _c(
        "P12.lifetime_s",
        8.07e-9,
        "s",
        "Pinnington1997",
        tag="verified",
        uncertainty=0.09e-9,
        note="gamma/2pi = 19.72 MHz total, 19.62 MHz partial to S1/2; Olmschenk's tighter 8.12(2) ns is "
        "P12.lifetime_olmschenk_s",
    ),
    _c(
        "P12.hfs_splitting_hz",
        2.105e9,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        note="F' = 1 above F' = 0 (= A for I = J = 1/2): the 2.1 GHz optical-pumping sideband and the detuning of the "
        "dominant detection error",
    ),
    _c(
        "P12.lifetime_olmschenk_s",
        8.12e-9,
        "s",
        "Olmschenk2007",
        tag="verified",
        uncertainty=0.02e-9,
        note="Olmschenk et al., Phys. Rev. A 80, 022502 (2009); a cross-check, 0.6% above the 8.07 ns the plan's "
        "anchors are computed from",
    ),
    _c(
        "P12.branching_to_D32",
        0.00501,
        "",
        "Olmschenk2007",
        tag="verified",
        uncertainty=0.00015,
        note="measured on 174Yb+, isotope independent; CONDITIONAL on tau(P1/2) = 8.07 ns because the fit measures "
        "the product gamma R; the channel is at 2.438 um, not at the 935.2 nm repump",
    ),
    _c("D32.energy_cm", 22960.80, "cm^-1", "NIST_ASD_5_12"),
    _c("D32.lifetime_s", 52.7e-3, "s", "Olmschenk2007", tag="verified"),
    _c(
        "D32.hfs_splitting_hz",
        0.86e9,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        uncertainty=0.02e9,
        note="F = 2 above F = 1; A = splitting/2 for J = 3/2, I = 1/2",
    ),
    _c(
        "D32.g_J",
        0.802,
        "",
        "Meggers1967",
        note="the observed Zeeman g as NIST ASD 5.12 lists it, corrected there from a printed 1.802; the Lande value is "
        "0.79954",
    ),
    _c("D52.energy_cm", 24332.69, "cm^-1", "NIST_ASD_5_12"),
    _c(
        "D52.g_J",
        1.202,
        "",
        "Meggers1967",
        note="the observed Zeeman g as NIST ASD 5.12 lists it; the Lande value is 1.20046",
    ),
    _c(
        "D52.lifetime_s",
        7.2e-3,
        "s",
        "Taylor1997",
        tag="verified",
        uncertainty=0.3e-3,
        note="measured in 172Yb+; the 171Yb+ per-F values 7.1(4) and 7.4(4) ms (Tan et al. 2021) agree",
    ),
    _c(
        "D52.A_hfs_hz",
        -63.368e6,
        "Hz",
        "Tan2021",
        tag="verified",
        uncertainty=0.001e6,
        note="INVERTED: the printed splitting is -190.104(3) MHz and A = splitting/3 for I = 1/2, J = 5/2",
    ),
    _c("P32.energy_cm", 30392.23, "cm^-1", "NIST_ASD_5_12"),
    _c(
        "P32.lifetime_s",
        6.15e-9,
        "s",
        "Pinnington1997",
        tag="verified",
        uncertainty=0.09e-9,
        note="the fine-structure partner of P1/2 from the same measurement; gamma/2pi = 25.883 MHz",
    ),
    _c(
        "P32.branching_to_D32",
        0.0017,
        "",
        "Feldker2018",
        tag="verified",
        uncertainty=0.0001,
        note="1345.6 nm; the three fractions of Table III sum to 1, so the S1/2 share is 1 - 0.0017 - 0.0108",
    ),
    _c(
        "P32.branching_to_D52",
        0.0108,
        "",
        "Feldker2018",
        tag="verified",
        uncertainty=0.0005,
        note="1650.3 nm; theory (Biemont et al. 1998) gives 98.77/0.21/1.02 %",
    ),
    _c(
        "P32.A_hfs_hz",
        875.4e6,
        "Hz",
        "Feldker2018",
        tag="verified",
        uncertainty=1.0e6,
        note="A > 0 since mu_I > 0 (F' = 2 above F' = 1)",
    ),
    _c(
        "P32.A_hfs_berends_hz",
        877e6,
        "Hz",
        "Berends1992",
        tag="verified",
        uncertainty=20e6,
        note="the first measurement, concordant at 0.1 sigma; a cross-check",
    ),
    _c(
        "F72.energy_cm",
        21418.75,
        "cm^-1",
        "NIST_ASD_5_12",
        note="the F7/2 trap state, cleared at 638.6 nm",
    ),
    _c(
        "F72.g_J",
        1.145,
        "",
        "Meggers1967",
        note="the observed Zeeman g as NIST ASD 5.12 lists it; the Lande value of the 4f13 6s2 2F7/2 hole is 1.14319",
    ),
    _c(
        "F72.lifetime_s",
        9.96e7,
        "s",
        "Lange2021",
        tag="corrected",
        uncertainty=0.50e7,
        note="3.16(16) yr: the published 1.58(8) yr used E_0^2 where <E^2> = E_0^2/2 belongs (arXiv:2107.11229v2)",
    ),
    _c(
        "F72.A_hfs_hz",
        905.0e6,
        "Hz",
        "Taylor1999",
        tag="verified",
        uncertainty=0.5e6,
        note="splitting 3620(2) MHz and A = splitting/4 for I = 1/2, J = 7/2",
    ),
    _c(
        "bracket_3D32_12.energy_cm",
        33653.86,
        "cm^-1",
        "NIST_ASD_5_12",
        note="4f13(2F7/2)5d6s 3[3/2]1/2, the 935.2 nm repump target",
    ),
    _c(
        "bracket_3D32_12.hfs_splitting_hz",
        2.2095e9,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        uncertainty=1.1e6,
        note="INVERTED (F = 0 above F = 1), so A < 0",
    ),
    _c(
        "bracket_3D32_12.g_J",
        1.320,
        "",
        "Meggers1967",
        note="the observed Zeeman g as NIST ASD 5.12 lists it; the level is jK-coupled and has no LS Lande value",
    ),
    _c(
        "bracket_3D32_12.lifetime_s",
        37.7e-9,
        "s",
        "Pinnington1994",
        tag="extracted",
        uncertainty=0.5e-9,
        note="read by secondary quotation; the 4.2 MHz linewidth of the 935 nm line (arXiv:2111.11504) corroborates "
        "it to 1%",
    ),
    _c(
        "bracket_3D32_12.A_297nm_per_s",
        2.61e7,
        "s^-1",
        "SansonettiMartin2005",
        note="Einstein A of the 297.143 nm line to S1/2 (NIST ASD T7227): 0.98397 of the decay, leaving 0.01603 for "
        "the 935.2 nm branch. Declared on the level as untabulated_branching rather than tabulated, so the S1/2 "
        "intermediate sums stay the P1/2 + P3/2 doublet",
    ),
    _c(
        "bracket_1D52_52.energy_cm",
        37077.59,
        "cm^-1",
        "NIST_ASD_5_12",
        note="4f13(2F7/2)5d6s 1[5/2]5/2, the 638.6 nm clear-out target",
    ),
    _c(
        "bracket_1D52_52.g_J",
        1.113,
        "",
        "Meggers1967",
        note="the observed Zeeman g as NIST ASD 5.12 lists it; the level is jK-coupled and has no LS Lande value",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """The ``Species`` record for 171Yb+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    nuclear_spin = Fraction(t[_P + "nuclear_spin"].value).limit_denominator(2)
    e_p12 = energy_hz(t[_P + "P12.energy_cm"])
    e_p32 = energy_hz(t[_P + "P32.energy_cm"])
    e_d32 = energy_hz(t[_P + "D32.energy_cm"])
    e_d52 = energy_hz(t[_P + "D52.energy_cm"])
    e_f72 = energy_hz(t[_P + "F72.energy_cm"])
    e_br = energy_hz(t[_P + "bracket_3D32_12.energy_cm"])

    s12 = Level(
        name="S1/2",
        energy_hz=0.0,
        lifetime_s=None,
        A_hfs_hz=a_hfs_from_two_manifold_splitting(
            t[_P + "S12.hfs_splitting_hz"], nuclear_spin, half, inverted=False
        ),
        B_hfs_hz=0.0,
        g_J=t[_P + "S12.g_J"].value,
        citations=("Olmschenk2007", "Han2025"),
    )
    p12 = Level(
        name="P1/2",
        energy_hz=e_p12,
        lifetime_s=t[_P + "P12.lifetime_s"].value,
        A_hfs_hz=a_hfs_from_two_manifold_splitting(
            t[_P + "P12.hfs_splitting_hz"], nuclear_spin, half, inverted=False
        ),
        B_hfs_hz=0.0,
        g_J=lande_g_j(1, half, half),
        citations=("NIST_ASD_5_12", "Pinnington1997", "Olmschenk2007", "PLAN_background"),
    )
    d32 = Level(
        name="D3/2",
        energy_hz=e_d32,
        lifetime_s=t[_P + "D32.lifetime_s"].value,
        A_hfs_hz=a_hfs_from_two_manifold_splitting(
            t[_P + "D32.hfs_splitting_hz"], nuclear_spin, Fraction(3, 2), inverted=False
        ),
        B_hfs_hz=0.0,
        g_J=t[_P + "D32.g_J"].value,
        citations=("NIST_ASD_5_12", "Olmschenk2007", "Meggers1967"),
    )
    p32 = Level(
        name="P3/2",
        energy_hz=e_p32,
        lifetime_s=t[_P + "P32.lifetime_s"].value,
        A_hfs_hz=t[_P + "P32.A_hfs_hz"].value,
        B_hfs_hz=0.0,  # I = 1/2 has no electric-quadrupole hyperfine term
        g_J=lande_g_j(1, half, Fraction(3, 2)),
        citations=("NIST_ASD_5_12", "Pinnington1997", "Feldker2018", "Berends1992", "PLAN_background"),
    )
    d52 = Level(
        name="D5/2",
        energy_hz=e_d52,
        lifetime_s=t[_P + "D52.lifetime_s"].value,
        A_hfs_hz=t[_P + "D52.A_hfs_hz"].value,
        B_hfs_hz=0.0,
        g_J=t[_P + "D52.g_J"].value,
        citations=("NIST_ASD_5_12", "Taylor1997", "Tan2021", "Meggers1967"),
    )
    f72 = Level(
        name="F7/2",
        energy_hz=e_f72,
        lifetime_s=t[_P + "F72.lifetime_s"].value,
        A_hfs_hz=t[_P + "F72.A_hfs_hz"].value,
        B_hfs_hz=0.0,
        g_J=t[_P + "F72.g_J"].value,
        citations=("NIST_ASD_5_12", "Lange2021", "Taylor1999", "Meggers1967"),
    )
    # the 297.143 nm 3[3/2]1/2 -> S1/2 channel carries A tau of the decay and is declared, not tabulated
    b_297 = t[_P + "bracket_3D32_12.A_297nm_per_s"].value * t[_P + "bracket_3D32_12.lifetime_s"].value
    bracket = Level(
        name="3D[3/2]1/2",
        energy_hz=e_br,
        lifetime_s=t[_P + "bracket_3D32_12.lifetime_s"].value,
        A_hfs_hz=a_hfs_from_two_manifold_splitting(
            t[_P + "bracket_3D32_12.hfs_splitting_hz"], nuclear_spin, half, inverted=True
        ),
        B_hfs_hz=0.0,
        g_J=t[_P + "bracket_3D32_12.g_J"].value,
        citations=("NIST_ASD_5_12", "Olmschenk2007", "Pinnington1994", "SansonettiMartin2005", "Meggers1967"),
        untabulated_branching=(("S1/2 at 297.143 nm (NIST ASD A = 2.61e7 s^-1)", b_297),),
    )

    gamma_p12 = gamma_hz_from_lifetime(t[_P + "P12.lifetime_s"])
    b_d = t[_P + "P12.branching_to_D32"].value
    cites = ("NIST_ASD_5_12", "Pinnington1997", "Olmschenk2007")
    s_p = Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), gamma_p12, 1.0 - b_d, "E1", cites)
    d_p = Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), gamma_p12, b_d, "E1", cites)

    gamma_p32 = gamma_hz_from_lifetime(t[_P + "P32.lifetime_s"])
    b32_d32 = t[_P + "P32.branching_to_D32"].value
    b32_d52 = t[_P + "P32.branching_to_D52"].value
    p32_cites = ("NIST_ASD_5_12", "Pinnington1997", "Feldker2018")
    s_p32 = Transition(
        "S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), gamma_p32, 1.0 - b32_d32 - b32_d52, "E1", p32_cites
    )
    d32_p32 = Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), gamma_p32, b32_d32, "E1", p32_cites)
    d52_p32 = Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), gamma_p32, b32_d52, "E1", p32_cites)
    d32_br = Transition(
        "D3/2",
        "3D[3/2]1/2",
        wavelength_vac_m(e_d32, e_br),
        gamma_hz_from_lifetime(t[_P + "bracket_3D32_12.lifetime_s"]),
        1.0 - b_297,
        "E1",
        ("NIST_ASD_5_12", "Pinnington1994", "SansonettiMartin2005", "Olmschenk2007"),
    )

    return Species(
        name=NAME,
        mass_u=ion_mass_u(t[_P + "mass_atomic_u"]),
        nuclear_spin=float(nuclear_spin),
        mu_I_nuclear_magnetons=t[_P + "mu_I_nuclear_magnetons"].value,
        levels=(s12, p12, p32, d32, d52, f72, bracket),
        transitions=(s_p, d_p, s_p32, d32_p32, d52_p32, d32_br),
        qubit=("S1/2 F=0 mF=0", "S1/2 F=1 mF=0"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-3D[3/2]1/2",),
        shelving=None,
    )
