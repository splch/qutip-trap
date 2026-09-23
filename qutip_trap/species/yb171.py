"""171Yb+ species table (I = 1/2).

Level energies are NIST ASD 5.12 values in cm^-1; the ground-state g_J is the calculated value of Han et al. 2025.
Excited-level g_J are rounded NIST ASD values, not measurements.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    Cited,
    MissingConstant,
    Tag,
    a_hfs_from_two_manifold_splitting,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)

NAME = "171Yb+"
_P = "yb171."


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
        170.9363302,
        "u",
        "NIST_AWIC",
        uncertainty=2.2e-6,
        note="relative atomic mass of the neutral atom; Species.mass_u subtracts one electron mass "
        "(170.93578 u, the ion mass PLAN.md 9.16 row 13-6 pins for the Lamb-Dicke anchor)",
    ),
    _c("nuclear_spin", 0.5, "hbar", "Olmschenk2007", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        0.49367,
        "mu_N",
        "PLAN_4_5_1",
        note="used by PLAN.md 4.5.1 and Section 13 for g_I in the 310.87 Hz/G^2 recomputation; the primary "
        "measurement is not named in the plan",
    ),
    _c(
        "S12.hfs_splitting_hz",
        12_642_812_118.5,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        note="ZERO-FIELD ground-state splitting, equal to A_hfs for I = J = 1/2; the 12.642815, 12.642819 and "
        "12.642821 GHz printed elsewhere are second-order-shifted values at 3.0 to 5.3 G and must never be "
        "stored as zero-field constants (PLAN.md 4.5.6)",
    ),
    _c(
        "S12.g_J",
        2.002615,
        "",
        "Han2025",
        tag="corrected",
        uncertainty=7e-5,
        note="adopted by PLAN.md 4.5.1 after the 2026-09-04 critique; NIST ASD lists the spectroscopic 1.998 "
        "and the plan's second revision carried three uncited values (2.00225664, 2.00254, 2.00292), all "
        "superseded; a theoretical determination with no modern measurement behind it (Section 12)",
    ),
    _c("P12.energy_cm", 27061.82, "cm^-1", "NIST_ASD_5_12"),
    _c(
        "P12.lifetime_s",
        8.07e-9,
        "s",
        "Pinnington1997",
        tag="verified",
        uncertainty=0.09e-9,
        note="gamma/2pi = 19.72 MHz total, 19.62 MHz partial to S1/2 (PLAN.md 9.12 '171Yb+ constants'); the "
        "19.6 and 21 MHz readings of the same line in earlier revisions are retired. The full locator behind "
        "PLAN.md 4.5.6's bare 'Pinnington et al.' is Pinnington, Rieger and Kernahan, Phys. Rev. A 56, 2421 "
        "(1997), which prints 8.07(9) ns for the j = 1/2 level and 6.15(9) ns for j = 3/2 in one abstract "
        "(audit item E25; the source key Pinnington_via_Olmschenk2007 is retired). Olmschenk et al. 2007's "
        "own 8.12(2) ns is the tighter modern value and is stored as a cross-check",
    ),
    _c(
        "P12.hfs_splitting_hz",
        2.105e9,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        note="F' = 1 above F' = 0; equals A_hfs for I = J = 1/2; the 2.1 GHz sideband of optical pumping "
        "(PLAN.md 4.2.6) and the detuning of the dominant detection error (8.1)",
    ),
    _c("P12.g_J", 0.667, "", "NIST_ASD_5_12", note="the Lande value 2/3; no measured g_J in the plan"),
    _c(
        "P12.lifetime_olmschenk_s",
        8.12e-9,
        "s",
        "Olmschenk2007",
        tag="verified",
        uncertainty=0.02e-9,
        note="the tighter modern P1/2 lifetime (Olmschenk et al., Phys. Rev. A 80, 022502 (2009)), 4.5x "
        "tighter than Pinnington's 8.07(9) ns and 0.6% above it; a cross-check, because every PLAN.md "
        "anchor (19.72 MHz, 19.62 MHz partial, 1.752 e a0, 50.83 mW/cm^2) is computed from the 8.07 ns",
    ),
    _c(
        "P12.branching_to_D32",
        0.00501,
        "",
        "Olmschenk2007",
        tag="verified",
        uncertainty=0.00015,
        note="measured on 174Yb+, isotope independent; CONDITIONAL on tau(P1/2) = 8.07 ns because the fit "
        "measures the product gamma*R (PLAN.md 4.5.6); the P1/2 -> D3/2 channel lies at 2.438 um, not at "
        "the 935.2 nm repump, a conflation that changes the inferred reduced element by 4.2 times",
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
    _c("D32.g_J", 0.802, "", "NIST_ASD_5_12"),
    _c("D52.energy_cm", 24332.69, "cm^-1", "NIST_ASD_5_12"),
    _c("D52.g_J", 1.202, "", "NIST_ASD_5_12"),
    _c(
        "D52.lifetime_s",
        7.2e-3,
        "s",
        "Taylor1997",
        tag="verified",
        uncertainty=0.3e-3,
        note="measured in 172Yb+; the 171Yb+ per-F values are 7.1(4) and 7.4(4) ms (Tan et al. 2021), "
        "consistent, so the isotope difference is below the quoted precision. The 411 nm E2 shelving line",
    ),
    _c(
        "D52.A_hfs_hz",
        -63.368e6,
        "Hz",
        "Tan2021",
        tag="verified",
        uncertainty=0.001e6,
        note="INVERTED (A < 0): the printed splitting is -190.104(3) MHz and A = splitting/3 for I = 1/2, "
        "J = 5/2. Stored as A directly rather than as a splitting because the source prints the sign",
    ),
    _c("P32.energy_cm", 30392.23, "cm^-1", "NIST_ASD_5_12"),
    _c("P32.g_J", 1.333, "", "NIST_ASD_5_12"),
    _c(
        "P32.lifetime_s",
        6.15e-9,
        "s",
        "Pinnington1997",
        tag="verified",
        uncertainty=0.09e-9,
        note="the fine-structure partner of the 8.07(9) ns P1/2 from the same beam-laser measurement; "
        "gamma/2pi = 25.883 MHz total. Without this level the Section 4.5.4 intermediate sum has one path "
        "instead of two, so Omega_R falls as 1/Delta instead of 1/Delta^2 (audit item E4)",
    ),
    _c(
        "P32.branching_to_D32",
        0.0017,
        "",
        "Feldker2018",
        tag="verified",
        uncertainty=0.0001,
        note="1345.6 nm; Table III. The three printed fractions 0.9875(6)/0.0017(1)/0.0108(5) sum to 1.0000, "
        "so the S1/2 share is taken as 1 - 0.0017 - 0.0108 = 0.9875 exactly",
    ),
    _c(
        "P32.branching_to_D52",
        0.0108,
        "",
        "Feldker2018",
        tag="verified",
        uncertainty=0.0005,
        note="1650.3 nm; Table III. Theory (Biemont et al. 1998) gives 98.77/0.21/1.02 %",
    ),
    _c(
        "P32.A_hfs_hz",
        875.4e6,
        "Hz",
        "Feldker2018",
        tag="verified",
        uncertainty=1.0e6,
        note="A > 0 because mu_I(171Yb) > 0 (F' = 2 above F' = 1); stored as A, not as the 2A splitting",
    ),
    _c(
        "P32.A_hfs_berends_hz",
        877e6,
        "Hz",
        "Berends1992",
        tag="verified",
        uncertainty=20e6,
        note="the first measurement, concordant with Feldker et al. 2018's 875.4(10) MHz at 0.1 sigma; a "
        "cross-check, never the value used",
    ),
    _c(
        "F72.energy_cm",
        21418.75,
        "cm^-1",
        "NIST_ASD_5_12",
        note="the F7/2 trap state, cleared at 638.6 nm (PLAN.md 4.5.6, 6.7); NIST levels give 638.62 nm",
    ),
    _c("F72.g_J", 1.145, "", "NIST_ASD_5_12"),
    _c(
        "F72.lifetime_s",
        9.96e7,
        "s",
        "Lange2021",
        tag="corrected",
        uncertainty=0.50e7,
        note="3.16(16) yr. The PUBLISHED 1.58(8) yr is superseded: arXiv:2107.11229v2 (14 May 2026) records "
        "that E_0^2 was used where <E^2> = E_0^2/2 belongs, so the lifetime was low by exactly a factor two",
    ),
    _c(
        "F72.A_hfs_hz",
        905.0e6,
        "Hz",
        "Taylor1999",
        tag="verified",
        uncertainty=0.5e6,
        note="splitting 3620(2) MHz and A = splitting/4 for I = 1/2, J = 7/2; A > 0 as for every other 171Yb+ "
        "level, mu_I being positive",
    ),
    _c(
        "bracket_3D32_12.energy_cm",
        33653.86,
        "cm^-1",
        "NIST_ASD_5_12",
        note="4f13(2F7/2)5d6s 3[3/2]1/2, the 935.2 nm repump target; NIST levels give 935.18 nm",
    ),
    _c(
        "bracket_3D32_12.hfs_splitting_hz",
        2.2095e9,
        "Hz",
        "Olmschenk2007",
        tag="verified",
        uncertainty=1.1e6,
        note="INVERTED multiplet (F = 0 above F = 1), so A < 0 (PLAN.md 4.5.6)",
    ),
    _c("bracket_3D32_12.g_J", 1.320, "", "NIST_ASD_5_12"),
    _c(
        "bracket_3D32_12.lifetime_s",
        37.7e-9,
        "s",
        "Pinnington1994",
        tag="extracted",
        uncertainty=0.5e-9,
        note="the 935.2 nm repump upper level. NOT primary-verified: the paper's table was not read (APS 403) "
        "and the digits reach this table by secondary quotation; the 4.2 MHz natural linewidth of the 935 nm "
        "line quoted in arXiv:2111.11504 corroborates it to 1% (1/(2 pi x 4.2 MHz) = 37.9 ns)",
    ),
    _c(
        "bracket_3D32_12.A_297nm_per_s",
        2.61e7,
        "s^-1",
        "SansonettiMartin2005",
        note="Einstein A of the 297.143 nm 3[3/2]1/2 -> S1/2 line (NIST ASD reference code T7227, no accuracy "
        "rating). With the 37.7 ns total lifetime this channel carries 0.98397 of the decay, leaving 0.01603 "
        "for the 935.2 nm branch that closes the repump cycle. The 297 nm line is deliberately NOT tabulated "
        "as a Transition: it would put a level whose lifetime is not primary-verified into every Section 4.5.4 "
        "intermediate sum out of S1/2, which the plan specifies as the P1/2 + P3/2 doublet. It is declared "
        "instead on the level as untabulated_branching, so the branching set still sums to 1",
    ),
    _c(
        "bracket_1D52_52.energy_cm",
        37077.59,
        "cm^-1",
        "NIST_ASD_5_12",
        note="4f13(2F7/2)5d6s 1[5/2]5/2, the 638.6 nm clear-out target",
    ),
    _c("bracket_1D52_52.g_J", 1.113, "", "NIST_ASD_5_12"),
)

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("1D[5/2]5/2 lifetime and hyperfine A (638.6 nm clear-out)", "not in PLAN.md"),
    MissingConstant(
        "D5/2 and F7/2 decay branchings (D5/2 -> S1/2 versus -> F7/2, measured at 17.6(4)/18.4(4)% per F by "
        "Feldker et al. 2018 and Tan et al. 2021; F7/2 -> S1/2 versus -> D5/2)",
        "Feldker et al., Phys. Rev. A 97, 032511 (2018); Tan et al., arXiv:2012.14187 -- read the tables before "
        "entering, and note the two levels then need E2/M1 Transition records",
    ),
    MissingConstant(
        "measured g_J of any 171Yb+ level except the ground state (the seven stored values are rounded NIST "
        "ASD literals, not measurements)",
        "NIST ASD forwards g_J to a primary reference; follow the chain or tag them [background] like 40Ca+",
    ),
)


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
        g_J=t[_P + "P12.g_J"].value,
        citations=("NIST_ASD_5_12", "Pinnington1997", "Olmschenk2007"),
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
        citations=("NIST_ASD_5_12", "Olmschenk2007"),
    )
    p32 = Level(
        name="P3/2",
        energy_hz=e_p32,
        lifetime_s=t[_P + "P32.lifetime_s"].value,
        A_hfs_hz=t[_P + "P32.A_hfs_hz"].value,
        B_hfs_hz=0.0,  # I = 1/2 has no electric-quadrupole hyperfine term
        g_J=t[_P + "P32.g_J"].value,
        citations=("NIST_ASD_5_12", "Pinnington1997", "Feldker2018", "Berends1992"),
    )
    d52 = Level(
        name="D5/2",
        energy_hz=e_d52,
        lifetime_s=t[_P + "D52.lifetime_s"].value,
        A_hfs_hz=t[_P + "D52.A_hfs_hz"].value,
        B_hfs_hz=0.0,
        g_J=t[_P + "D52.g_J"].value,
        citations=("NIST_ASD_5_12", "Taylor1997", "Tan2021"),
    )
    f72 = Level(
        name="F7/2",
        energy_hz=e_f72,
        lifetime_s=t[_P + "F72.lifetime_s"].value,
        A_hfs_hz=t[_P + "F72.A_hfs_hz"].value,
        B_hfs_hz=0.0,
        g_J=t[_P + "F72.g_J"].value,
        citations=("NIST_ASD_5_12", "Lange2021", "Taylor1999"),
    )
    # the 297 nm 3[3/2]1/2 -> S1/2 channel (A tau of the decay) is declared, not tabulated, so the branchings sum to 1
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
        citations=("NIST_ASD_5_12", "Olmschenk2007", "Pinnington1994", "SansonettiMartin2005"),
        untabulated_branching=(("S1/2 at 297.143 nm (NIST ASD A = 2.61e7 s^-1)", b_297),),
    )

    gamma_p12 = gamma_hz_from_lifetime(t[_P + "P12.lifetime_s"])
    b_d = t[_P + "P12.branching_to_D32"].value
    cites = ("NIST_ASD_5_12", "Pinnington1997", "Olmschenk2007")
    s_p = Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), gamma_p12, 1.0 - b_d, "E1", cites)
    d_p = Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), gamma_p12, b_d, "E1", cites)

    # the P3/2 doublet partner, the second path of the Raman intermediate sum
    gamma_p32 = gamma_hz_from_lifetime(t[_P + "P32.lifetime_s"])
    b32_d32 = t[_P + "P32.branching_to_D32"].value
    b32_d52 = t[_P + "P32.branching_to_D52"].value
    p32_cites = ("NIST_ASD_5_12", "Pinnington1997", "Feldker2018")
    s_p32 = Transition(
        "S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), gamma_p32, 1.0 - b32_d32 - b32_d52, "E1", p32_cites
    )
    d32_p32 = Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), gamma_p32, b32_d32, "E1", p32_cites)
    d52_p32 = Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), gamma_p32, b32_d52, "E1", p32_cites)
    # the 935.2 nm repump line
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
