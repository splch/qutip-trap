"""The per-channel omega^3 of the decay amplitudes, the branching-normalization invariant, and the 171Yb+
fine-structure doublet (PLAN.md 4.5.5, 4.5.4; Section 9.16 row 4.5-7; milestone M0a, audit items E1-E4, E8).

These are the tests that would have caught defect B1 of the 2026-09-07 audit: every scattering fixture in
``tests/atomic_fixtures.py`` gives each P level exactly ONE lower level, where the normalization of
:meth:`AtomicStructure.decay_amplitudes` is unconditionally 1 whatever the weight, so the missing per-channel
omega^3 was invisible. On a real multi-channel species it overstated 171Yb+'s P1/2 -> D3/2 leakage by 118x.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.light.beams import Beam
from qutip_trap.species import species
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import C_M_PER_S, TWO_PI

B_GAUSS = 4.0
Z_HAT = (0.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def yb() -> Species:
    return species("171Yb+")


@pytest.fixture(scope="module")
def ca() -> Species:
    return species("40Ca+")


# ---- E1 + E3: |c|^2 is the partial-rate fraction on a real multi-channel species ------------------------


@pytest.mark.parametrize("name", ["171Yb+", "40Ca+", "88Sr+"])
def test_decay_amplitude_shares_reproduce_the_tabulated_branchings(name: str) -> None:
    """sum_{b in lo, q} |c_{e->b q}|^2 / Gamma_e == Transition.branching(lo -> e), to 1e-12, for every
    E1-reached upper level of every buildable species and every one of its dressed sublevels.

    PLAN.md 4.5.5 fixes <b|d_q'|e> = sqrt(3 pi eps0 hbar c^3 Gamma_e/omega_e^3) c_{e->b,q'}, so |c|^2 is the
    partial-rate fraction and carries omega_{e,b}^3 PER CHANNEL. Weighting the raw elements by
    omega^(3/2) before normalizing is what makes this identity hold; without it the weight per channel is
    Gamma_partial/omega^3 and 171Yb+ P1/2's shares come out 0.4087/0.5913 instead of 0.99499/0.00501.
    """
    sp = species(name)
    st = AtomicStructure(sp, B_GAUSS, Z_HAT)
    checked = 0
    for level in {up for (_lo, up) in st.e1}:
        gamma_e = st.total_decay_rate_rad_s(level)
        want = {lo: st.e1[(lo, level)].branching for lo in st.lower_levels_of(level)}
        for e in st.states_of(level):
            got: dict[str, float] = dict.fromkeys(want, 0.0)
            for (label, _q), amp in st.decay_amplitudes(e).items():
                got[label.split()[0]] += abs(amp) ** 2 / gamma_e
            for lo, fraction in want.items():
                assert got[lo] == pytest.approx(fraction, rel=1e-12, abs=1e-15), (
                    f"{name} {e.full_label} -> {lo}"
                )
            checked += 1
    assert checked >= 4, f"{name}: no multi-sublevel upper level was exercised"


def test_the_omega_cubed_weight_is_what_reproduces_them(yb: Species) -> None:
    """The negative control for the fix: dropping the per-channel omega^(3/2) weight (normalizing the bare
    elements, which is what the code did before 2026-09-07) overstates 171Yb+'s D3/2 share by 118x."""
    st = AtomicStructure(yb, B_GAUSS, Z_HAT)
    e = st.states_of("P1/2")[0]
    unweighted: dict[str, float] = {}
    for lo in st.lower_levels_of("P1/2"):
        for b in st.states_of(lo):
            for q in (-1, 0, 1):
                val = st._lower_upper_element(b, e, q)
                if val != 0.0:
                    unweighted[lo] = unweighted.get(lo, 0.0) + abs(val) ** 2
    total = sum(unweighted.values())
    shares = {lo: v / total for lo, v in unweighted.items()}
    assert shares["D3/2"] == pytest.approx(0.591306, rel=1e-4)
    assert shares["S1/2"] == pytest.approx(0.408694, rel=1e-4)
    assert shares["D3/2"] / yb.transition("D3/2-P1/2").branching == pytest.approx(118.03, rel=1e-3)


def test_the_scattering_budget_of_the_355_nm_drive_leaks_at_the_tabulated_rate(yb: Species) -> None:
    """The consequence the audit measured: on the real 355 nm clock drive the leakage share out of the qubit
    manifold is the tabulated 0.501% + 0.17%/1.08% D branchings weighted by the two paths, not 59%."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    beam = Beam(
        wavelength_m=355e-9,
        power_w=10e-3,
        waist_m=20e-6,
        k_hat=(1.0, 0.0, 0.0),
        polarization=(0.0, 1.0, 0.0),
        pointing_m=(0.0, 0.0, 0.0),
    )
    rates = st.scattering_rates(st.state("S1/2 F=0 mF=0"), beam)
    total = sum(rates.values())
    leak = sum(v for k, v in rates.items() if not k.startswith("S1/2"))
    assert 0.0 < leak / total < 0.02, "the D-state leakage share is percent-level, not 59%"
    assert sum(v for k, v in rates.items() if k.startswith("S1/2")) / total > 0.98


# ---- E2: the branching normalization has teeth ----------------------------------------------------------


def _two_channel_species(branchings: tuple[float, float], deficit: float | None) -> Species:
    """A toy S/D/P species whose P level has two tabulated E1 channels of the given branchings."""
    e_d = 1.0e13
    e_p = 8.0e14
    tau = 8.0e-9
    gamma = 1.0 / (TWO_PI * tau)
    untab = () if deficit is None else (("an untabulated channel", deficit),)
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D3/2", e_d, 1.0, 0.0, 0.0, 0.8, ("test",)),
        Level("P1/2", e_p, tau, 0.0, 0.0, 0.667, ("test",), untabulated_branching=untab),
    )
    trs = (
        Transition("S1/2", "P1/2", C_M_PER_S / e_p, gamma, branchings[0], "E1", ("test",)),
        Transition("D3/2", "P1/2", C_M_PER_S / (e_p - e_d), gamma, branchings[1], "E1", ("test",)),
    )
    return Species(
        name="toy",
        mass_u=40.0,
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=levels,
        transitions=trs,
        qubit=("S1/2 mJ=-1/2", "D3/2 mJ=-1/2"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2",),
        shelving=None,
    )


def test_a_complete_branching_set_builds() -> None:
    sp = _two_channel_species((0.94, 0.06), None)
    assert sp.level("P1/2").untabulated_branching == ()


def test_an_incomplete_branching_set_is_refused_unless_the_deficit_is_declared() -> None:
    """Defect B4: ``raman.py`` used to renormalize over whatever channels happened to be tabulated, so 40Ca+
    P3/2's 5.9% of D-state leakage silently became 100% cycling. Now the Species refuses to build."""
    with pytest.raises(ValueError, match=r"E1 branchings out of P1/2 sum to .* not 1 within 1e-9"):
        _two_channel_species((0.90, 0.06), None)
    # ... and building it with the deficit DECLARED is allowed, at the declared size
    sp = _two_channel_species((0.90, 0.06), 0.04)
    assert sp.level("P1/2").untabulated_branching == (("an untabulated channel", 0.04),)


def test_a_declared_deficit_scales_the_amplitudes_rather_than_being_renormalized_away() -> None:
    """sum_{b q} |c|^2 equals the TABULATED total, so a declared 4% leak out of the manifold stays 4% missing
    instead of being redistributed over the tabulated channels."""
    sp = _two_channel_species((0.90, 0.06), 0.04)
    st = AtomicStructure(sp, B_GAUSS, Z_HAT)
    assert st.tabulated_branching_total("P1/2") == pytest.approx(0.96, rel=1e-12)
    e = st.states_of("P1/2")[0]
    gamma = st.total_decay_rate_rad_s("P1/2")
    got = sum(abs(a) ** 2 for a in st.decay_amplitudes(e).values()) / gamma
    assert got == pytest.approx(0.96, rel=1e-12)
    shares: dict[str, float] = {}
    for (label, _q), amp in st.decay_amplitudes(e).items():
        shares[label.split()[0]] = shares.get(label.split()[0], 0.0) + abs(amp) ** 2 / gamma
    assert shares["S1/2"] == pytest.approx(0.90, rel=1e-12)
    assert shares["D3/2"] == pytest.approx(0.06, rel=1e-12)


def test_a_declared_deficit_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        Level("P1/2", 1.0, 1e-8, 0.0, 0.0, 0.667, ("test",), untabulated_branching=(("x", 1.5),))
    with pytest.raises(ValueError, match="must name itself"):
        Level("P1/2", 1.0, 1e-8, 0.0, 0.0, 0.667, ("test",), untabulated_branching=(("", 0.5),))


# ---- E8: one total decay rate per upper level -----------------------------------------------------------


def test_transitions_out_of_one_upper_level_must_agree_on_its_total_rate() -> None:
    """Defect B5: ``total_decay_rate_rad_s`` used ``set.pop()``, so two disagreeing gamma_hz returned an
    arbitrary one. The invariant is now enforced at construction (a level with lifetime_s = None and two
    inconsistent transitions used to pass validation)."""
    e_d, e_p, gamma = 1.0e13, 8.0e14, 2.0e7
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D3/2", e_d, 1.0, 0.0, 0.0, 0.8, ("test",)),
        Level("P1/2", e_p, None, 0.0, 0.0, 0.667, ("test",)),  # no lifetime: the old escape hatch
    )
    trs = (
        Transition("S1/2", "P1/2", C_M_PER_S / e_p, gamma, 0.94, "E1", ("test",)),
        Transition("D3/2", "P1/2", C_M_PER_S / (e_p - e_d), 1.5 * gamma, 0.06, "E1", ("test",)),
    )
    with pytest.raises(ValueError, match="disagree on its total decay rate"):
        Species(
            name="toy",
            mass_u=40.0,
            nuclear_spin=0.0,
            mu_I_nuclear_magnetons=0.0,
            levels=levels,
            transitions=trs,
            qubit=("S1/2 mJ=-1/2", "D3/2 mJ=-1/2"),
            cycling="S1/2-P1/2",
            repumps=("D3/2-P1/2",),
            shelving=None,
        )


# ---- E4: the 171Yb+ fine-structure doublet --------------------------------------------------------------


def test_yb171_now_carries_the_p32_doublet_partner(yb: Species) -> None:
    """PLAN.md 4.5.4 requires the intermediate sum over "all hyperfine and Zeeman sublevels of the P1/2 AND
    P3/2 manifolds"; before 2026-09-07 the record had only P1/2 (defect B2)."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    assert sorted(st.upper_levels_of("S1/2")) == ["P1/2", "P3/2"]
    p32 = yb.level("P3/2")
    assert p32.lifetime_s == 6.15e-9  # Pinnington, Rieger, Kernahan 1997
    assert p32.A_hfs_hz == 875.4e6  # Feldker et al. 2018
    assert yb.transition("S1/2-P3/2").branching == pytest.approx(0.9875, abs=1e-12)
    assert yb.transition("D3/2-P3/2").branching == 0.0017
    assert yb.transition("D5/2-P3/2").branching == 0.0108
    assert len(st.states_of("P3/2")) == 8  # F' = 1 and F' = 2


def _lin_perp_lin(lam: float) -> tuple[Beam, Beam]:
    """Two counter-aligned linear polarizations perpendicular to B = z: pure sigma content, no pi."""
    common = {"power_w": 1e-3, "waist_m": 20e-6, "pointing_m": (0.0, 0.0, 0.0), "k_hat": (0.0, 0.0, 1.0)}
    return (
        Beam(wavelength_m=lam, polarization=(1.0, 0.0, 0.0), **common),  # type: ignore[arg-type]
        Beam(wavelength_m=lam, polarization=(0.0, 1.0, 0.0), **common),  # type: ignore[arg-type]
    )


def test_the_two_paths_enter_with_opposite_signs(yb: Species) -> None:
    """PLAN.md 4.5.4: "the two paths enter with opposite signs, which is the origin of the
    omega_f/[Delta(Delta - omega_f)] structure and of the 1/Delta^2 fall-off"."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    for detuning_thz in (-50.0, -100.0, -200.0, -400.0):
        lam = C_M_PER_S / (f0 + detuning_thz * 1e12)
        b1, b2 = _lin_perp_lin(lam)
        c1 = {e.full_label: (om, d) for e, om, d in st.couplings_from(dn, b1)}
        c2 = {e.full_label: om for e, om, _ in st.couplings_from(up, b2)}
        parts: dict[str, complex] = {}
        for key, (om1, d1) in c1.items():
            om2 = c2.get(key)
            if om2 is not None:
                lv = key.split()[0]
                parts[lv] = parts.get(lv, 0.0 + 0.0j) + np.conj(om2) * om1 / (2.0 * d1)
        assert set(parts) == {"P1/2", "P3/2"}
        ratio = parts["P3/2"] / parts["P1/2"]
        assert ratio.real < 0.0, f"the two paths must interfere destructively at {detuning_thz} THz"
        assert abs(ratio.imag) < 1e-9 * abs(ratio.real), "the two partial sums are collinear in phase"


def test_the_raman_coupling_follows_the_two_path_closed_form_not_one_over_delta(yb: Species) -> None:
    """|Omega_R| |Delta(Delta - omega_f)|/omega_f is flat where |Omega_R Delta| is not.

    171Yb+'s fine structure is 99.84 THz, so the asymptotic 1/Delta^2 regime (|Delta| >> omega_f) is not
    physically reachable at optical frequencies -- the beam would have to leave the optical band. The
    discriminator against the single-path form is therefore the closed form itself: over Delta/2pi = -50 to
    -400 THz the two-path invariant varies by 27% while the single-path one varies by a factor 2.6.
    """
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    omega_f_thz = (yb.level("P3/2").energy_hz - yb.level("P1/2").energy_hz) / 1e12
    assert omega_f_thz == pytest.approx(99.8432, rel=1e-5)
    two_path: list[float] = []
    one_path: list[float] = []
    for detuning_thz in (-50.0, -100.0, -200.0, -400.0):
        lam = C_M_PER_S / (f0 + detuning_thz * 1e12)
        b1, b2 = _lin_perp_lin(lam)
        omega = abs(st.raman_coupling_rad_s(dn, up, b1, b2))
        two_path.append(omega * abs(detuning_thz * (detuning_thz - omega_f_thz)) / omega_f_thz)
        one_path.append(omega * abs(detuning_thz))
    spread = max(two_path) / min(two_path)
    assert spread < 1.35, f"the two-path invariant should be flat, got a spread of {spread}"
    assert max(one_path) / min(one_path) > 2.5, "the single-path form must NOT be flat"


def test_the_raman_spin_flip_rate_is_interference_protected_and_the_leakage_is_not(yb: Species) -> None:
    """PLAN.md 4.5.5: the Raman rate is protected by the destructive interference and falls much faster than
    the D-state leakage, which "is not protected by the interference" (the eps_D floor of PLAN.md 457)."""
    st = AtomicStructure(yb, 5.0, Z_HAT)
    dn, up = st.state("S1/2 F=0 mF=0"), st.state("S1/2 F=1 mF=0")
    f0 = yb.level("P1/2").energy_hz
    flips: list[float] = []
    leaks: list[float] = []
    for detuning_thz in (-100.0, -200.0, -400.0):
        lam = C_M_PER_S / (f0 + detuning_thz * 1e12)
        b1, _ = _lin_perp_lin(lam)
        rates = st.scattering_rates(dn, b1)
        flips.append(rates.get(up.full_label, 0.0))
        leaks.append(sum(v for k, v in rates.items() if not k.startswith("S1/2")))
    flip_decades = [flips[i + 1] / flips[i] for i in range(2)]
    leak_decades = [leaks[i + 1] / leaks[i] for i in range(2)]
    assert all(r < 0.1 for r in flip_decades), (
        f"the spin-flip rate must fall faster than 1/Delta^3: {flip_decades}"
    )
    assert all(0.2 < r < 0.45 for r in leak_decades), (
        f"the leakage must fall as about 1/Delta^2: {leak_decades}"
    )


def test_the_935_nm_repump_line_is_a_tabulated_transition(yb: Species) -> None:
    """Defect B6: a designated repump used to be validated against level NAMES only, so
    species("171Yb+").transition("D3/2-3D[3/2]1/2") raised KeyError."""
    tr = yb.transition("D3/2-3D[3/2]1/2")
    assert tr.multipole == "E1"
    assert tr.wavelength_vac_m == pytest.approx(935.186e-9, rel=1e-5)
    # the 297.143 nm channel is DECLARED, not tabulated, so the branchings still sum to 1
    declared = yb.level("3D[3/2]1/2").untabulated_branching
    assert len(declared) == 1 and "297" in declared[0][0]
    assert tr.branching + declared[0][1] == pytest.approx(1.0, abs=1e-12)
    assert tr.branching == pytest.approx(0.01603, rel=1e-3)


def test_the_ca40_854_and_850_nm_lines_are_tabulated_with_the_gerritsma_split(ca: Species) -> None:
    """Defect B6 for 40Ca+: the designated 854 nm D5/2 repump had no Transition record and P3/2's decay was
    renormalized into the 393 nm cycling line at 100%. Gerritsma et al. 2008 supply the split."""
    assert ca.transition("D5/2-P3/2").branching == 0.0587
    assert ca.transition("D3/2-P3/2").branching == 0.00661
    assert ca.transition("S1/2-P3/2").branching == pytest.approx(0.93469, abs=1e-12)
    assert ca.transition("D5/2-P3/2").wavelength_vac_m == pytest.approx(854.444e-9, rel=1e-5)
    assert ca.transition("D3/2-P3/2").wavelength_vac_m == pytest.approx(850.036e-9, rel=1e-5)
    st = AtomicStructure(ca, B_GAUSS, Z_HAT)
    assert st.tabulated_branching_total("P3/2") == pytest.approx(1.0, abs=1e-12)


def test_the_sr88_cycling_and_repump_lines_are_tabulated(ca: Species) -> None:
    """Defects B6 and B7: 88Sr+ was "available" with the E2 clock line as its ONLY transition, so its
    designated 421.7 nm cycling line and 1092 nm repump raised KeyError and every Raman, light-shift and
    scattering quantity returned 0 with no error."""
    sr = species("88Sr+")
    st = AtomicStructure(sr, B_GAUSS, Z_HAT)
    assert sorted(st.upper_levels_of("S1/2")) == ["P1/2", "P3/2"]
    assert sr.transition(sr.cycling).wavelength_vac_m == pytest.approx(421.671e-9, rel=1e-5)
    for label in sr.repumps:
        assert sr.transition(label).multipole == "E1"
    assert sr.transition("D5/2-P3/2").wavelength_vac_m == pytest.approx(1033.014e-9, rel=1e-5)
    # E13b: the 674 nm clock wavelength now comes from the stored clock FREQUENCY, not the level energy
    assert sr.transition("S1/2-D5/2").wavelength_vac_m == pytest.approx(674.025591e-9, rel=1e-9)


def test_atomic_structure_refuses_a_species_with_no_e1_structure() -> None:
    """E7: an E2-only record cannot answer any Section 4.5.4/4.5.5 question, so building the structure on one
    must raise instead of returning zeros."""
    levels = (
        Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("test",)),
        Level("D5/2", 4.4e14, 0.39, 0.0, 0.0, 1.2, ("test",)),
    )
    trs = (Transition("S1/2", "D5/2", C_M_PER_S / 4.4e14, 1.0 / (TWO_PI * 0.39), 1.0, "E2", ("test",)),)
    e2_only = Species(
        name="e2only",
        mass_u=88.0,
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=levels,
        transitions=trs,
        qubit=("S1/2 mJ=-1/2", "D5/2 mJ=-1/2"),
        cycling="S1/2-D5/2",
        repumps=(),
        shelving="S1/2-D5/2",
    )
    # the E2 pair IS the qubit, so the structure builds and says so in its approximations
    st = AtomicStructure(e2_only, B_GAUSS, Z_HAT)
    assert st.e1 == {}
    with pytest.raises(KeyError, match="no tabulated E1"):
        st.reduced_element_c_m("S1/2", "D5/2")
    with pytest.raises(ValueError, match="no E1"):
        AtomicStructure(
            Species(
                name="e2only-e1qubit",
                mass_u=88.0,
                nuclear_spin=0.0,
                mu_I_nuclear_magnetons=0.0,
                levels=levels,
                transitions=trs,
                qubit=("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"),
                cycling="S1/2-D5/2",
                repumps=(),
                shelving="S1/2-D5/2",
            ),
            B_GAUSS,
            Z_HAT,
        )


def test_the_partial_rates_of_a_multi_channel_level_sum_to_the_total(yb: Species) -> None:
    """The identity behind the fix, at the Transition level: sum_lo partial_rate = gamma_rad_s."""
    for level in ("P1/2", "P3/2"):
        trs = [t for t in yb.transitions if t.upper == level and t.multipole == "E1"]
        total = sum(t.partial_rate_rad_s for t in trs)
        assert total == pytest.approx(trs[0].gamma_rad_s, rel=1e-12)
        assert math.isclose(1.0 / (TWO_PI * yb.level(level).lifetime_s or 1.0), trs[0].gamma_hz, rel_tol=1e-9)


# ---- E16: the far-detuning diagnostic -------------------------------------------------------------------


def test_the_far_detuning_diagnostic_reports_and_refuses(yb: Species) -> None:
    """PLAN.md 4.5.6: the second-order sums "have no i gamma/2 in their denominators and are used only far
    from resonance, the multi-level Bloch solve of Section 4.2.8 taking over within a few linewidths".

    Before 2026-09-07 nothing said which side of that line a call was on: ``raman.py`` divided by Delta_e
    with no check, so a beam resonant with a dressed intermediate sublevel returned ``inf`` rather than a
    diagnostic (defect B14).
    """
    st = AtomicStructure(yb, 5.0, Z_HAT)
    a = st.state("S1/2 F=0 mF=0")
    far = Beam(
        wavelength_m=355e-9,
        power_w=1e-3,
        waist_m=20e-6,
        k_hat=(1.0, 0.0, 0.0),
        polarization=(0.0, 1.0, 0.0),
        pointing_m=(0.0, 0.0, 0.0),
    )
    margin = st.nearest_resonance_in_linewidths(a, far)
    # 355 nm is 33.19 THz from P1/2, whose Gamma/2pi is 19.72 MHz: 1.68e6 linewidths (the P3/2 path is
    # 66.65 THz from a 25.88 MHz line, 2.58e6 linewidths, so P1/2 is the nearer resonance)
    assert margin == pytest.approx(1.683e6, rel=0.02)
    st.refuse_if_near_resonance(a, far)  # does not raise
    e = st.states_of("P1/2")[0]
    resonant = Beam(
        wavelength_m=C_M_PER_S / (e.energy_hz - a.energy_hz),
        power_w=1e-9,
        waist_m=20e-6,
        k_hat=(1.0, 0.0, 0.0),
        polarization=(0.0, 1.0, 0.0),
        pointing_m=(0.0, 0.0, 0.0),
    )
    assert st.nearest_resonance_in_linewidths(a, resonant) < 1e-6
    with pytest.raises(ValueError, match="linewidths from a dressed intermediate sublevel"):
        st.refuse_if_near_resonance(a, resonant)
