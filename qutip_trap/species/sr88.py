"""88Sr+ species table (PLAN.md Sections 4.5.7, 9.14, 12; Appendix E).

I = 0 (Lande g_J, [background]). The 674 nm quadrupole record is complete: the D5/2 lifetime 390.8(1.6) ms
of Letchumanan et al. 2005 (read from the thesis, corroborated by Jiang 2009 and the NIST Sr II compilation),
the clock frequency fixing lambda_vac = 674.025591 nm, and the LS-estimated 9e-5 M1 branch to D3/2. The
P-level lifetimes and branchings (422, 408, 1092, 1033 nm) are not in PLAN.md and are listed as missing; the
``Species`` builds because its transition labels only need the levels to exist.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    MissingConstant,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

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
        note="674.0256 nm vacuum; matches the clock frequency to 1e-7",
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
        note="NIST Sr II compilation, accuracy AA; = 1/0.3908 s",
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
        "D52.lifetime_theory_jiang_s",
        0.394,
        "s",
        "Jiang2009",
        tag="verified",
        uncertainty=0.003,
        note="theory; bracket only",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant(
        "P1/2 and P3/2 lifetimes and their branchings into S1/2, D3/2, D5/2 (422, 408, 1092, 1033 nm)",
        "not in PLAN.md",
    ),
    MissingConstant("D3/2 lifetime", "not in PLAN.md"),
    MissingConstant("measured g_J values (Lande used)", "not in PLAN.md"),
)


def species() -> Species:
    """The Appendix E ``Species`` record for 88Sr+: complete on the 674 nm quadrupole line, P levels without rates."""
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t["sr88.D32.energy_cm"])
    e_d52 = energy_hz(t["sr88.D52.energy_cm"])
    e_p12 = energy_hz(t["sr88.P12.energy_cm"])
    e_p32 = energy_hz(t["sr88.P32.energy_cm"])
    lande = ("NIST_ASD_5_12", "PLAN_background")
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
        Level("P1/2", e_p12, None, 0.0, 0.0, lande_g_j(1, half, half), lande),
        Level("P3/2", e_p32, None, 0.0, 0.0, lande_g_j(1, half, Fraction(3, 2)), lande),
    )
    s_d52 = Transition(
        "S1/2",
        "D5/2",
        wavelength_vac_m(0.0, e_d52),
        gamma_hz_from_lifetime(t["sr88.D52.lifetime_s"]),
        1.0 - t["sr88.D52.M1_branch_to_D32"].value,
        "E2",
        ("NIST_ASD_5_12", "Letchumanan2005", "Sansonetti2012", "PLAN_background"),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["sr88.mass_atomic_u"]),
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=levels,
        transitions=(s_d52,),
        qubit=("S1/2 mJ=-1/2", "D5/2 mJ=-1/2"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-D5/2",
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
