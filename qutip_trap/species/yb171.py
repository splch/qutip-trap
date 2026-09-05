"""171Yb+ species table (PLAN.md Sections 4.5.6, 8.1, 9.12, 9.13, 9.15; Appendix E).

Every number is a :class:`~qutip_trap.provenance.Cited` constant with its source and provenance tag;
nothing hyperfine-resolved is typed in (Section 4.5.6: the module "shrinks to a table of cited constants").
Level energies are NIST ASD 5.12 values in cm^-1; the hyperfine constants, lifetimes and branchings are
the Run 4 extractions from Olmschenk et al. 2007; g_J of the ground state is the adopted Han et al. 2025
value. The conversions applied at ingest are those of :mod:`qutip_trap.species.table` only.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited, Tag
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    MissingConstant,
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
        ledger_id=_P + suffix,
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
        "Pinnington_via_Olmschenk2007",
        tag="verified",
        uncertainty=0.09e-9,
        note="gamma/2pi = 19.72 MHz total, 19.62 MHz partial to S1/2 (PLAN.md 9.12 '171Yb+ constants'); the "
        "19.6 and 21 MHz readings of the same line in earlier revisions are retired",
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
    _c("P32.energy_cm", 30392.23, "cm^-1", "NIST_ASD_5_12"),
    _c("P32.g_J", 1.333, "", "NIST_ASD_5_12"),
    _c(
        "F72.energy_cm",
        21418.75,
        "cm^-1",
        "NIST_ASD_5_12",
        note="the F7/2 trap state, cleared at 638.6 nm (PLAN.md 4.5.6, 6.7); NIST levels give 638.62 nm",
    ),
    _c("F72.g_J", 1.145, "", "NIST_ASD_5_12"),
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
        "bracket_1D52_52.energy_cm",
        37077.59,
        "cm^-1",
        "NIST_ASD_5_12",
        note="4f13(2F7/2)5d6s 1[5/2]5/2, the 638.6 nm clear-out target",
    ),
    _c("bracket_1D52_52.g_J", 1.113, "", "NIST_ASD_5_12"),
    _c(
        "repump_935_sideband_hz",
        3.07e9,
        "Hz",
        "Olmschenk2007",
        note="sideband on the 935.2 nm repump that addresses both D3/2 hyperfine components; essential for "
        "optical pumping (PLAN.md 4.2.6, 8.1)",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant(
        "D5/2 lifetime and hyperfine A (411 nm E2 line)", "not in PLAN.md; Olmschenk 2007 or NIST ASD lines"
    ),
    MissingConstant("F7/2 lifetime and hyperfine A", "not in PLAN.md"),
    MissingConstant(
        "3D[3/2]1/2 lifetime and its branching into S1/2 versus D3/2 (closes the 935 nm repump cycle)",
        "not in PLAN.md",
    ),
    MissingConstant("1D[5/2]5/2 lifetime and hyperfine A (638.6 nm clear-out)", "not in PLAN.md"),
    MissingConstant("P3/2 lifetime, branching and hyperfine A (329 nm)", "not in PLAN.md"),
)


def species() -> Species:
    """The Appendix E ``Species`` record for 171Yb+, built from :data:`TABLE` alone."""
    t = TABLE
    half = Fraction(1, 2)
    nuclear_spin = Fraction(t[_P + "nuclear_spin"].value).limit_denominator(2)
    e_p12 = energy_hz(t[_P + "P12.energy_cm"])
    e_d32 = energy_hz(t[_P + "D32.energy_cm"])
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
        citations=("NIST_ASD_5_12", "Pinnington_via_Olmschenk2007", "Olmschenk2007"),
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
    bracket = Level(
        name="3D[3/2]1/2",
        energy_hz=e_br,
        lifetime_s=None,
        A_hfs_hz=a_hfs_from_two_manifold_splitting(
            t[_P + "bracket_3D32_12.hfs_splitting_hz"], nuclear_spin, half, inverted=True
        ),
        B_hfs_hz=0.0,
        g_J=t[_P + "bracket_3D32_12.g_J"].value,
        citations=("NIST_ASD_5_12", "Olmschenk2007"),
    )

    gamma_p12 = gamma_hz_from_lifetime(t[_P + "P12.lifetime_s"])
    b_d = t[_P + "P12.branching_to_D32"].value
    cites = ("NIST_ASD_5_12", "Pinnington_via_Olmschenk2007", "Olmschenk2007")
    s_p = Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), gamma_p12, 1.0 - b_d, "E1", cites)
    d_p = Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), gamma_p12, b_d, "E1", cites)

    return Species(
        name=NAME,
        mass_u=ion_mass_u(t[_P + "mass_atomic_u"]),
        nuclear_spin=float(nuclear_spin),
        mu_I_nuclear_magnetons=t[_P + "mu_I_nuclear_magnetons"].value,
        levels=(s12, p12, d32, bracket),
        transitions=(s_p, d_p),
        qubit=("S1/2 F=0 mF=0", "S1/2 F=1 mF=0"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-3D[3/2]1/2",),
        shelving=None,
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
