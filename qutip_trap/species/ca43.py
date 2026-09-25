"""43Ca+ species table (PLAN.md Section 4.5.6): 40Ca+'s fine structure with the 43Ca hyperfine
structure (I = 7/2, mu_I < 0, so every A is negative and F = 3 lies above F = 4).

The level energies, P lifetimes, branchings and D lifetimes are isotope-independent and read from the 40Ca+ table (the
~1 GHz isotope shift is below the 0.01 cm^-1 its energies resolve). The g_J of S1/2 and D5/2 are measurements on 40Ca+
applied as isotope-independent; the P levels and D3/2 have no measured g_J (theory: Sahoo, arXiv:1710.06558) and carry
the Lande values ([background]).
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.species import ca40
from qutip_trap.species.model import Species
from qutip_trap.species.table import CitedFactory, ion_mass_u

NAME = "43Ca+"
_c = CitedFactory("ca43.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 42.95876644, "u", "NIST_AWIC", uncertainty=2.4e-7),
    _c("nuclear_spin", 3.5, "hbar", "Harty2014", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.31535,
        "mu_N",
        "Hanley2021",
        tag="verified",
        uncertainty=9e-6,
        note="the FREE-ION moment (uncorrected for the bound electrons' shielding), the one a trapped-ion Breit-Rabi "
        "Hamiltonian takes; Stone's bare-nucleus -1.31733(6) mu_N is a different quantity",
    ),
    _c(
        "S12.A_hfs_hz",
        -806.4020716e6,
        "Hz",
        "Arbes1994",
        tag="verified",
        uncertainty=8e-2,
        note="zero-field splitting 3225.608 MHz = |A|(I + 1/2), SIGNED negative (inverted multiplet); with A > 0 "
        "the clock point exists nowhere between 1 and 4000 G",
    ),
    _c(
        "S12.g_J",
        2.00225664,
        "",
        "Tommaseo2003",
        tag="extracted",
        uncertainty=9e-8,
        note="measured on 40Ca+ in a Penning trap and applied as isotope-independent (as Hanley et al., "
        "arXiv:2105.10352, do in their 43Ca+ fit); the digits are read in Hanley et al.",
    ),
    _c(
        "P12.A_hfs_hz",
        -145.4e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=0.1e6,
    ),
    _c(
        "P32.A_hfs_hz",
        -31.0e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=0.2e6,
        note="-31.0(2) MHz (Garcia Ruiz et al., Phys. Rev. C 91, 041304(R) (2015) Table I); the -31.43(19) MHz that "
        "circulates is the 45Ca value",
    ),
    _c("P32.B_hfs_hz", -6.9e6, "Hz", "Nortershauser1998", tag="verified", uncertainty=1.7e6),
    _c("D32.A_hfs_hz", -47.3e6, "Hz", "Nortershauser1998", tag="verified", uncertainty=0.2e6),
    _c("D32.B_hfs_hz", -3.7e6, "Hz", "Nortershauser1998", tag="verified", uncertainty=1.9e6),
    _c(
        "D52.A_hfs_hz",
        -3.8931e6,
        "Hz",
        "Benhelm2007",
        tag="verified",
        uncertainty=0.0002e6,
        note="the signs are the erratum's, Phys. Rev. A 75, 049901(E); arXiv v1/v2 print both positive",
    ),
    _c("D52.B_hfs_hz", -4.241e6, "Hz", "Benhelm2007", tag="verified", uncertainty=0.004e6),
    _c(
        "D52.g_J",
        1.2003340,
        "",
        "Chwalla2009",
        tag="verified",
        uncertainty=3e-7,
        note="measured on 40Ca+, isotope-independent; the Lande value is 1.2004639",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        -0.0408,
        "b",
        "Stone2019",
        tag="verified",
        uncertainty=0.0008,
        note="Q(43Ca) < 0, why every B above is negative",
    ),
    _c(
        "P.linewidth_ozeri_hz",
        22.5e6,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I gamma/2pi for Ca+",
    ),
    _c(
        "S12.hfs_splitting_ozeri_hz", 3.23e9, "Hz", "Ozeri2007", tag="verified", note="Ozeri Table I, rounded"
    ),
    _c(
        "fine_structure_splitting_hz",
        6.68e12,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_f/2pi; the NIST levels give 6.682 THz",
    ),
    _c("wavelength_ozeri_397_m", 396.8e-9, "m", "Ozeri2007", tag="verified", note="an AIR wavelength"),
    _c("wavelength_ozeri_393_m", 393.4e-9, "m", "Ozeri2007", tag="verified", note="an AIR wavelength"),
    _c("P_to_D_branching_inverse", 17.0, "", "Ozeri2007", tag="verified", note="Ozeri Table I f^-1"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """The ``Species`` record for 43Ca+: the 40Ca+ levels and E1 lines with this table's hyperfine constants and g_J."""
    t = TABLE
    ca = ca40.species()
    lv = {level.name: level for level in ca.levels}
    levels = (
        replace(
            lv["S1/2"],
            A_hfs_hz=t["ca43.S12.A_hfs_hz"].value,
            g_J=t["ca43.S12.g_J"].value,
            citations=("Arbes1994", "Tommaseo2003", "Hanley2021"),
        ),
        replace(
            lv["D3/2"],
            A_hfs_hz=t["ca43.D32.A_hfs_hz"].value,
            B_hfs_hz=t["ca43.D32.B_hfs_hz"].value,
            citations=("NIST_ASD_5_12", "Nortershauser1998", "Kreuter2005", "PLAN_background"),
        ),
        replace(
            lv["D5/2"],
            A_hfs_hz=t["ca43.D52.A_hfs_hz"].value,
            B_hfs_hz=t["ca43.D52.B_hfs_hz"].value,
            g_J=t["ca43.D52.g_J"].value,
            citations=("NIST_ASD_5_12", "Benhelm2007", "Chwalla2009", "Kreuter2005"),
        ),
        replace(
            lv["P1/2"],
            A_hfs_hz=t["ca43.P12.A_hfs_hz"].value,
            citations=("NIST_ASD_5_12", "Nortershauser1998", "Hettrich2015", "PLAN_background"),
        ),
        replace(
            lv["P3/2"],
            A_hfs_hz=t["ca43.P32.A_hfs_hz"].value,
            B_hfs_hz=t["ca43.P32.B_hfs_hz"].value,
            citations=("NIST_ASD_5_12", "Nortershauser1998", "Meir2020", "PLAN_background"),
        ),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["ca43.mass_atomic_u"]),
        nuclear_spin=float(Fraction(t["ca43.nuclear_spin"].value).limit_denominator(2)),
        mu_I_nuclear_magnetons=t["ca43.mu_I_nuclear_magnetons"].value,
        levels=levels,
        transitions=tuple(tr for tr in ca.transitions if tr.multipole == "E1"),
        # the |4,0> <-> |3,+1> clock qubit at 146.0942 G, 397 and 866 nm Doppler cooling, 393/850/854 nm readout and
        # reset, M-resolving 393 nm shelving
        qubit=("S1/2 F=4 mF=0", "S1/2 F=3 mF=1"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-P3/2",
    )
