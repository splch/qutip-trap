"""Scaling I (PLAN.md M9a; Sections 5.1, 5.1.1, 5.2, 5.5, 9.8 row 3, 9.9, 9.17, 11.3): the contribution criterion's rows, the
frozen spectators' off-resonant bound, the ENR option with its sum-generator operator and marginals, the adaptive cap and
margin policy, the schedule start time and the spaces over a subset of the ions that GATE_LOCAL builds."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule, single_qubit_pulse
from qutip_trap.control.shaping import GateModes
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import build_hamiltonian
from qutip_trap.hilbert.operators import displacement_matrix_analytic, required_margin
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.hilbert.truncation import regrid_state
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.run.space import (
    DETUNING_GUARD_FACTOR,
    ModeContribution,
    classify,
    drive_operator_nonzeros,
    frozen_excitation_bounds,
    waveform_contributions,
)
from qutip_trap.units import TWO_PI
from tests.m4_fixtures import chain_device, derived_seeds, raman_gate_drives, table_with_waveform

OPTS = SolverOptions()

# ---- the contribution criterion (Sections 5.2, 11.3 item 2; Section 9.8 row 3; Section 9.17 "Freeze against drop") ----------


def test_a_weak_mode_two_kilohertz_from_a_tone_is_kept() -> None:
    """Section 9.8 row 3 / Section 11.3: a mode with eta = 1e-3 two kilohertz from a tone is kept (its residual displacement is
    far above the drop pair), while the same eta a megahertz away is dropped: the criterion is the closed-form contribution,
    never eta alone."""
    omega_gate = TWO_PI * 3.0e6
    probe = GateModes(
        ions=(0, 1),
        modes=(0, 1),
        omega_rad_s=(omega_gate, TWO_PI * 5.0e6),
        eta={0: (0.08, 1e-3), 1: (0.08, 1e-3)},
        nbar=(0.0, 0.0),
    )
    wf0 = Waveform.symmetric(probe, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False)
    assert wf0.segments is not None
    mu_tone = abs(float(wf0.segments[0].detuning_hz["blue"]))  # type: ignore[arg-type]
    # Section 5.2 line 822: the contribution pair decides alone, so a coupled mode below (1e-6, 1e-4) is DROPPED, not frozen
    # (|alpha|^2 (2n+1) = 8.1e-35 and |chi| = 2.8e-6 a megahertz away: nothing for a Debye-Waller factor to absorb)
    for gap_hz, expected in ((2e3, "resolved"), (1.0e6, "dropped")):
        modes = GateModes(
            ions=(0, 1),
            modes=(0, 1),
            omega_rad_s=(omega_gate, TWO_PI * (mu_tone + gap_hz)),
            eta={0: (0.08, 1e-3), 1: (0.08, 1e-3)},
            nbar=(0.0, 0.0),
        )
        wf = Waveform.symmetric(modes, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False)
        contrib = waveform_contributions(wf, modes, (0, 1))
        cls = classify(
            contrib[1],
            coupled=True,
            freeze_alpha_max=OPTS.freeze_alpha_max,
            freeze_chi_max_rad=OPTS.freeze_chi_max_rad,
        )
        assert cls == expected, (gap_hz, contrib[1], cls)
        if gap_hz < 1e5:
            assert contrib[1].alpha2_weighted > 1e-4, (
                "|alpha|^2 (2n + 1) of the near mode sits far above the drop threshold"
            )
    # against eta alone (Section 11.3): the eta = 1e-3 mode two kilohertz from the tone contributes MORE than an eta = 0.05
    # mode a megahertz away, which the criterion freezes (its chi_m of a few milliradians is absorbed by the calibration)
    near = GateModes(
        ions=(0, 1),
        modes=(0, 1),
        omega_rad_s=(omega_gate, TWO_PI * (mu_tone + 2e3)),
        eta={0: (0.08, 1e-3), 1: (0.08, 1e-3)},
        nbar=(0.0, 0.0),
    )
    far = GateModes(
        ions=(0, 1),
        modes=(0, 1),
        omega_rad_s=(omega_gate, TWO_PI * (mu_tone + 1e6)),
        eta={0: (0.08, 0.05), 1: (0.08, 0.05)},
        nbar=(0.0, 0.0),
    )
    c_near = waveform_contributions(
        Waveform.symmetric(near, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False),
        near,
        (0, 1),
    )
    c_far = waveform_contributions(
        Waveform.symmetric(far, gate_mode=0, epsilon_hz=20e3, duration_s=100e-6, all_modes=False), far, (0, 1)
    )
    assert c_near[1].alpha2_weighted > 100.0 * c_far[1].alpha2_weighted
    assert classify(c_far[1], coupled=True, freeze_alpha_max=1e-4, freeze_chi_max_rad=0.05) == "frozen"
    assert c_far[1].alpha2_weighted < 1e-6 and 1e-4 < c_far[1].chi_rad < 0.05


def test_freeze_against_drop_rows_of_section_9_17() -> None:
    """A spectator with |chi_m| = 0.03 rad and |alpha_m|^2 (2 nbar + 1) = 1e-5 is frozen, not dropped; one with 1e-7 and 1e-5 rad
    is dropped; one above the freeze tolerance in either quantity is resolved."""
    kw = dict(
        coupled=True, freeze_alpha_max=OPTS.freeze_alpha_max, freeze_chi_max_rad=OPTS.freeze_chi_max_rad
    )
    assert classify(ModeContribution(0, 1e-5, 0.03, 0.01, 0.05), **kw) == "frozen"
    # below the drop pair: dropped whatever couples to the mode (Section 5.2's criterion is the contribution pair alone;
    # intersecting the drop test with `coupled` made the dropped class unreachable, since a contribution only ever exists
    # for a coupled mode, and left `dropped_contribution` identically zero)
    assert (
        classify(
            ModeContribution(0, 1e-7, 1e-5, 0.01, 0.05),
            coupled=False,
            freeze_alpha_max=1e-4,
            freeze_chi_max_rad=0.05,
        )
        == "dropped"
    )
    assert classify(ModeContribution(0, 1e-7, 1e-5, 0.01, 0.05), **kw) == "dropped"
    assert classify(ModeContribution(0, 1e-5, 0.06, 0.01, 0.05), **kw) == "resolved"
    assert classify(ModeContribution(0, 2e-4, 0.001, 0.01, 0.05), **kw) == "resolved"
    assert classify(None, coupled=True, freeze_alpha_max=1e-4, freeze_chi_max_rad=0.05) == "frozen"
    assert classify(None, coupled=False, freeze_alpha_max=1e-4, freeze_chi_max_rad=0.05) == "dropped"


# ---- the frozen spectators' off-resonant excitation bound (Section 5.2) ------------------------------------------------------------


def test_frozen_excitation_bound_and_detuning_guard() -> None:
    """(eta Omega sqrt(n + 1)/(mu - l omega))^2 over the nearest sidebands per tone and ion; the guard notes a gap below
    20 eta Omega sqrt(n + 1); a tone on a sideband reports an unbounded excitation."""
    dev = chain_device(2)
    rabi, _stark = derived_seeds(dev, raman_gate_drives(2))
    omega_hz = rabi[(0, 0)]
    mu_hz = 3.0e6 + 20e3
    drive = Drive(
        kind="raman",
        ions=(0,),
        tones=(Tone(detuning_hz=mu_hz, phase_rad=0.0, envelope_hz=omega_hz),),
        beams=(0, 1),
        stark_shift_hz=0.0,
        crosstalk={},
    )
    pulse = Pulse(drive, 0.0, 50e-6, "p", ())
    from qutip_trap.light.raman import lamb_dicke_parameters

    etas, _ = lamb_dicke_parameters(dev, 0, drive.delta_k(dev.beams))
    bounds, guard = frozen_excitation_bounds(dev, [pulse], [2], {2: 0.5})
    w2 = dev.crystal.modes[2].omega_rad_s
    coupling = abs(etas[2]) * TWO_PI * omega_hz * math.sqrt(1.5)
    mu = TWO_PI * mu_hz
    expected = (coupling / abs(mu - w2)) ** 2 + (coupling / abs(mu + w2)) ** 2
    assert bounds[2] == pytest.approx(expected, rel=1e-12)
    assert bounds[2] > 1e-3, "the rocking mode 190 kHz from the tone is not a frozen candidate by this bound"
    # the guard: the gap 192 kHz against 20 eta Omega sqrt(n + 1)
    gap = abs(mu - w2)
    if gap <= DETUNING_GUARD_FACTOR * coupling:
        assert any("guard" in g for g in guard)
    else:
        assert not guard
    # a mode with eta = 0 contributes nothing; a tone exactly on a sideband is unbounded
    b0, _ = frozen_excitation_bounds(dev, [pulse], [0], {0: 3.0})
    assert b0[0] == 0.0
    on = Pulse(
        Drive("raman", (0,), (Tone(dev.crystal.modes[3].omega_hz, 0.0, omega_hz),), (0, 1), 0.0, {}),
        0.0,
        50e-6,
        "on",
        (),
    )
    b_on, g_on = frozen_excitation_bounds(dev, [on], [3], {3: 0.0})
    assert math.isinf(b_on[3]) and any("sits on sideband" in g for g in g_on)


# ---- the ENR option (Sections 5.1, 5.1.1, 11.3 item 1; Section 9.17 "ENR marginal") -------------------------------------------


def _embed_enr_state(space: HilbertSpace, ket: qt.Qobj, d_full: int) -> qt.Qobj:
    """The ENR-space ket as a ket on the product space (2, d_full, d_full) over the group's two modes."""
    dims, n_exc = space._enr_dims()
    _n, _s2i, i2s = qt.enr_state_dictionaries(dims, n_exc)
    arr = np.asarray(ket.full()).reshape(space.dims)
    out = np.zeros((space.ion_dims[0], d_full, d_full), dtype=complex)
    for idx in range(len(i2s)):
        n1, n2 = i2s[idx]
        out[:, n1, n2] = arr[:, idx]
    return qt.Qobj(out.reshape(-1, 1), dims=[[space.ion_dims[0], d_full, d_full], [1, 1, 1]])


def test_enr_marginal_equals_ptrace_of_the_product_space_state_and_the_shape_rule() -> None:
    """Section 9.17: ``HilbertSpace.marginal`` on an ENR factor equals ``ptrace`` of the same state embedded in the product space,
    and ``tensor(sigmap(), D_enr)`` reports dims multiplying to 98 where the shape is 56 for two modes at N_exc = 6."""
    space = HilbertSpace((2,), (), ((1, 2), 6), (0,))
    assert space.dims == [2, 28] and space.dimension == 56
    d_enr = space.enr_displacement({1: 0.1, 2: 0.05})
    op = qt.tensor(qt.sigmap(), d_enr)
    assert op.shape == (56, 56) and int(np.prod(op.dims[0])) == 98
    rng = np.random.default_rng(3)
    v = rng.normal(size=56) + 1j * rng.normal(size=56)
    v /= np.linalg.norm(v)
    ket = qt.Qobj(v.reshape(-1, 1), dims=[space.dims, [1, 1]])
    full = _embed_enr_state(space, ket, 7)
    for mode, factor in ((1, 1), (2, 2)):
        ours = space.mode_marginal(ket, mode)
        ref = full.ptrace(factor)
        assert (ours - ref).norm() < 1e-12
    assert (space.internal_marginal(ket) - full.ptrace(0)).norm() < 1e-12


def test_enr_displacement_is_the_sum_generator_exponential_and_agrees_with_the_product_far_from_the_cap() -> (
    None
):
    """Section 5.1.1: the ENR operator is expm of the SUM generator, unitary to 1e-15, equal to the product of per-mode
    displacements on the block n1 + n2 <= N_exc/2 to the table's precision and different near the cap."""
    space = HilbertSpace((2,), (), ((1, 2), 10), (0,))
    eta1, eta2 = 0.1, 0.05
    d_enr = space.enr_displacement({1: eta1, 2: eta2})
    assert (d_enr.dag() * d_enr - space.enr_identity()).norm() < 1e-12
    dims, n_exc = space._enr_dims()
    _n, s2i, _i2s = qt.enr_state_dictionaries(dims, n_exc)
    dense = np.asarray(d_enr.full())
    a1 = displacement_matrix_analytic(n_exc + 1, 1j * eta1)
    a2 = displacement_matrix_analytic(n_exc + 1, 1j * eta2)
    worst_inside = 0.0
    worst_cap = 0.0
    for (n1, n2), i in s2i.items():
        for (m1, m2), j in s2i.items():
            product = a1[m1, n1] * a2[m2, n2]
            diff = abs(dense[j, i] - product)
            if n1 + n2 <= n_exc // 2 and m1 + m2 <= n_exc // 2:
                worst_inside = max(worst_inside, diff)
            if n1 + n2 == n_exc or m1 + m2 == n_exc:
                worst_cap = max(worst_cap, diff)
    # PLAN.md 5.1.1's table gives 6e-16 for N_exc = 10 at eta = 0.1 and check_scaling.out prints 1.4e-15 here: the
    # operator-identity bar of 1e-12, not the 1e-9 M9a first asserted (M9a audit P1-8)
    assert worst_inside < 1e-12, worst_inside
    assert worst_cap > 1e-6, (
        "near the cap the sum-generator exponential is a different operator from the product"
    )


def test_enr_regrid_grows_the_excitation_cap_and_keeps_the_state() -> None:
    old = HilbertSpace((2,), (ModeTruncation(0, 5, (0, 1), 0.1),), ((1, 2), 4), ())
    new = old.grown(1, 2)
    assert new.enr_group == ((1, 2), 6) and new.dims == [2, 5, 28]
    st = old.initial_state([1], fock={0: 2, 1: 1, 2: 2})
    assert st.joint is not None
    big = regrid_state(st.joint, old, new)
    assert big.shape[0] == new.dimension and abs(big.norm() - 1.0) < 1e-12
    assert new.fock_populations(big, 1)[1] == pytest.approx(1.0) and new.fock_populations(big, 2)[
        2
    ] == pytest.approx(1.0)
    assert new.fock_populations(big, 0)[2] == pytest.approx(1.0)
    rho = qt.ket2dm(st.joint)
    big_rho = regrid_state(rho, old, new)
    assert (
        big_rho.tr() == pytest.approx(1.0)
        and (new.mode_marginal(big_rho, 2) - new.mode_marginal(big, 2)).norm() < 1e-12
    )
    with pytest.raises(ValueError):
        regrid_state(big, new, old)
    assert drive_operator_nonzeros(old) == 1 * 2 * 25 * 15**2
    assert (
        drive_operator_nonzeros(HilbertSpace((2, 2), (ModeTruncation(0, 4, (0, 1), 0.1),), None, ()))
        == 2 * 4 * 16
    )


# ---- the adaptive cap and margin policy (Sections 5.1.1, 5.5) ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    dev = chain_device(2)
    rabi, stark = derived_seeds(dev, raman_gate_drives(2))
    return dev, rabi, stark


def test_margin_policy_grows_a_cap_whose_margin_is_below_the_table_and_reports_the_range(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 5.5: a carrier pulse on a mode whose cap leaves fewer levels above the populated range than the Section 5.1.1
    margin for its eta is rerun with the cap raised by the deficit; the report states the populated range and the margin
    reached; ``margin_check=False`` keeps the cap and leaves the check to the boundary monitor alone."""
    dev, rabi, _stark = two_ion
    drives = raman_gate_drives(2)
    pulse = single_qubit_pulse(
        0, math.pi / 2.0, 0.0, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
    )
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    eta_com = 0.078
    need = required_margin(eta_com)
    # a cap of 5 levels above |0>: margin 4 < the 6 levels the table requires at eta <= 0.1
    space = HilbertSpace((2, 2), (ModeTruncation(3, 5, (0, 0), 0.1),), None, (0, 1, 2, 4, 5))
    state = space.initial_state([0, 0], fock={3: 0})
    engine = JointExactEngine()
    tr = engine.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions())
    rep = engine.last_report
    assert rep is not None and rep.growth_retries >= 1
    assert rep.space.truncation(3).d >= 1 + need, rep.space.dims
    assert (
        rep.populated_n_max[3] <= 1 and rep.margin_reached[3] >= need
    )  # the carrier leaks (eta Omega/nu)^2 into n = 1
    assert tr.final.joint is not None and tr.final.joint.shape[0] == rep.space.dimension
    engine_off = JointExactEngine()
    tr_off = engine_off.run_pulses(
        dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions(margin_check=False)
    )
    rep_off = engine_off.last_report
    assert rep_off is not None and rep_off.space == space and rep_off.growth_retries == 0
    # the populations agree: the carrier barely moves the mode, so the small cap was numerically fine here
    p_on = np.real(np.diag(tr.final.internal.full()))
    p_off = np.real(np.diag(tr_off.final.internal.full()))
    assert np.max(np.abs(p_on - p_off)) < 1e-6


def test_schedule_start_time_is_where_the_state_is_given(two_ion) -> None:  # type: ignore[no-untyped-def]
    """``Schedule.t0_s``: a GATE_LOCAL step hands the engine a state at the step's start; without it the engine evolves from
    min(0, the first start) as M2 to M8 did, which rotates a Fock component by e^{-i n omega t0} more."""
    dev, _rabi, _stark = two_ion
    t0, tau = 300e-6, 20e-6
    space = HilbertSpace((2, 2), (ModeTruncation(3, 8, (0, 2), 0.1),), None, (0, 1, 2, 4, 5))
    state = space.initial_state([0, 1], states={3: (qt.basis(8, 0) + qt.basis(8, 1)).unit()})
    a = JointExactEngine().run_pulses(
        dev,
        Schedule((), ((t0, t0 + tau),), (), {}, t0_s=t0),
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    b = JointExactEngine().run_pulses(
        dev,
        Schedule((), ((t0, t0 + tau),), (), {}),
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    assert a.times_s[0] == pytest.approx(t0) and b.times_s[0] == 0.0
    rho_a = space.mode_marginal(a.final.joint, 3)
    rho_b = space.mode_marginal(b.final.joint, 3)
    omega = dev.crystal.modes[3].omega_rad_s
    phase_a = np.angle(rho_a[1, 0])
    phase_b = np.angle(rho_b[1, 0])
    assert (phase_a - (-omega * tau)) % (2 * math.pi) == pytest.approx(0.0, abs=1e-6) or (
        phase_a - (-omega * tau)
    ) % (2 * math.pi) == pytest.approx(2 * math.pi, abs=1e-6)
    extra = (phase_b - phase_a + omega * t0) % (2 * math.pi)
    assert min(extra, 2 * math.pi - extra) < 1e-5, (
        "the whole-schedule convention rotates the coherence by omega t0 more"
    )


# ---- spaces over a subset of the ions (Section 5.4; the GATE_LOCAL building block) ------------------------------------------------


def test_local_space_over_a_subset_of_the_ions_reproduces_the_full_marginal() -> None:
    """A three-ion chain, a GPi2 pulse on ion 2 with crosstalk onto ion 1: the space over ions (0, 2) alone gives the same reduced
    state of (0, 2) as the three-ion space (the crosstalk onto ion 1 cannot change it), drops the crosstalk with a note, and its
    operators act on the right factor."""
    dev = chain_device(3)
    drives = raman_gate_drives(3)
    rabi, _stark = derived_seeds(dev, drives)
    pulse = single_qubit_pulse(
        2,
        math.pi / 2.0,
        0.3,
        drives[2],
        rabi[(2, 0)],
        0.0,
        gate_id="gpi2",
        crosstalk={1: 0.02},
        programmed=False,
    )
    sched = Schedule((pulse,), (), (), {})
    n_modes = len(dev.crystal.modes)
    local = HilbertSpace((2, 2), (), None, tuple(range(n_modes)), ions=(0, 2))
    assert (
        local.ion_labels == (0, 2) and local.ion_factor(2) == 1 and local.has_ion(2) and not local.has_ion(1)
    )
    with pytest.raises(KeyError):
        local.ion_factor(1)
    sp = local.sigma_plus(2)
    ref = qt.tensor(qt.qeye(2), qt.sigmap().dag().dag())  # |1><0| on the second factor
    assert (
        sp - qt.tensor(qt.qeye(2), qt.basis(2, 1) * qt.basis(2, 0).dag())
    ).norm() < 1e-14 and sp.shape == ref.shape
    built = build_hamiltonian(dev, [pulse], local, sample=quiet_sample())
    assert any("crosstalk" in a and "ion 1 outside the space dropped" in a for a in built.approximations)
    state_local = local.initial_state([1, 0])
    full = HilbertSpace((2, 2, 2), (), None, tuple(range(n_modes)))
    state_full = full.initial_state([1, 0, 0])
    opts = SolverOptions()
    tr_l = JointExactEngine().run_pulses(dev, sched, state_local, local, quiet_sample(), SeedSpec(0), opts)
    tr_f = JointExactEngine().run_pulses(dev, sched, state_full, full, quiet_sample(), SeedSpec(0), opts)
    rho_local = tr_l.final.internal
    rho_full_02 = full.marginal(tr_f.final.joint, (0, 2))
    assert (rho_local - rho_full_02).norm() < 1e-8
    # a pi/2 carrier pulse reduced by the frozen modes' Debye-Waller factors e^{-eta^2/2} (Section 5.2): a few permille below 1/2
    assert tr_l.expectations["P1[2]"][-1] == pytest.approx(0.5, abs=0.01)
    assert tr_l.expectations["P1[2]"][-1] < 0.5
    assert "P1[1]" not in tr_l.expectations and "P1[2]" in tr_l.expectations
    with pytest.raises(ValueError, match="does not carry"):
        build_hamiltonian(dev, [pulse], HilbertSpace((2,), (), None, tuple(range(n_modes)), ions=(0,)))
    with pytest.raises(ValueError, match="one per factor"):
        HilbertSpace((2, 2), (), None, (), ions=(0,))


# ---- frozen spectators against the joint run (Section 9.9) ---------------------------------------------------------------------


@pytest.mark.slow
def test_frozen_spectator_run_reproduces_the_joint_run_within_the_reported_bound() -> None:
    """Section 9.9: on the tilted-beam fixture the y-COM (2.9 MHz, 120 kHz below the tone, eta = 0.008) is a genuine frozen
    spectator of the x-COM gate; the frozen-spectator calibration and the joint one (the mode resolved) both reach chi = pi/4,
    and the final populations of the calibrated gates agree within the bound the frozen run reports: the mode's residual
    displacement, its off-resonant excitation and its chi_m loss the calibration absorbed."""
    from qutip_trap.calibration.entangling import calibrate_entangling_angle, exact_gate_check
    from qutip_trap.control.shaping import gate_modes
    from tests.m9_fixtures import Y_MODES_TWO_IONS, tilted_pair_device

    dev = tilted_pair_device(math.radians(6.0))
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(dev, drives)
    y_rock, y_com = Y_MODES_TWO_IONS
    modes = gate_modes(dev, (0, 1), (0, 1), nbar={m: 0.0 for m in range(6)})
    wf = Waveform.symmetric(modes, gate_mode=3, epsilon_hz=20e3, all_modes=True)
    contrib = waveform_contributions(wf, modes, (0, 1))
    assert (
        classify(contrib[y_com], coupled=True, freeze_alpha_max=1e-4, freeze_chi_max_rad=0.05) == "frozen"
        and 1e-4 < contrib[y_com].chi_rad < 0.05
    )
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    x_caps = (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13))
    frozen_space = HilbertSpace((2, 2), x_caps, None, (0, 1, y_rock, y_com))
    joint_space = HilbertSpace(
        (2, 2), x_caps + (ModeTruncation(y_com, 5, (0, 1), 0.02),), None, (0, 1, y_rock)
    )
    opts = SolverOptions(margin_check=False)
    runs = {}
    for name, space in (("frozen", frozen_space), ("joint", joint_space)):
        runs[name] = calibrate_entangling_angle(
            dev, wf, (0, 1), drives, table, space=space, tolerance_rad=2e-4, options=opts
        )
        assert runs[name].converged
    chi_f, chi_j = runs["frozen"].checks[-1].chi_rad, runs["joint"].checks[-1].chi_rad
    assert abs(chi_f - math.pi / 4.0) < 2e-4 and abs(chi_j - math.pi / 4.0) < 2e-4
    # the frozen run's bound: the frozen mode's residual displacement, off-resonant excitation and absorbed chi_m
    sched_pulses = list(
        __import__("qutip_trap.calibration.entangling", fromlist=["ms_schedule"])
        .ms_schedule(runs["frozen"].waveform, (0, 1), drives, table)
        .pulses
    )
    excitation, _guard = frozen_excitation_bounds(dev, sched_pulses, [y_com], {y_com: 0.0})
    bound = contrib[y_com].alpha2_weighted + excitation[y_com] + contrib[y_com].chi_rad
    # play each calibrated waveform on its own space from |00>: the populations agree within the bound
    pops = {}
    for name, space in (("frozen", frozen_space), ("joint", joint_space)):
        check, _tr = exact_gate_check(
            dev, runs[name].waveform, (0, 1), drives, table, space=space, options=opts
        )
        pops[name] = check.populations
    for key in ("P00", "P11", "P01", "P10"):
        assert abs(pops["frozen"][key] - pops["joint"][key]) < bound, (key, pops, bound)
    # and the joint gate played on the frozen run's amplitudes differs by the absorbed chi_m at most
    cross, _tr = exact_gate_check(
        dev, runs["frozen"].waveform, (0, 1), drives, table, space=joint_space, options=opts
    )
    assert abs(cross.chi_rad - chi_f) < 2.0 * contrib[y_com].chi_rad + 1e-3
