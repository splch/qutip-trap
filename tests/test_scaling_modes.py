"""Scaling (PLAN.md Sections 5.1, 5.1.1, 5.2, 5.3, 5.5, 9.8 row 3, 9.9, 9.17, 11.1, 11.3, 11.5): the contribution criterion's
rows, the frozen spectators' off-resonant bound, the ENR option, the cap and margin policy and its ceiling, the Section
11.5 guards on the declared space, the propagator cache's fingerprint, the schedule start time, spaces over a subset of the
ions, the convergence regime of Section 9.9, the integrator ladder and the trajectory sampling."""

from __future__ import annotations

import dataclasses
import math
import re
import warnings

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation, compile_to_native
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule, entangling_pulses, single_qubit_pulse
from qutip_trap.control.schedule import schedule as make_schedule
from qutip_trap.control.shaping import GateModes, gate_modes, symmetric_pulse, waveform_integrals
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions, TruncationLimit
from qutip_trap.dynamics.evolve import LARGE_MODE_DIMENSION, evolve
from qutip_trap.dynamics.hamiltonian import build_hamiltonian
from qutip_trap.dynamics.operators import displacement_matrix_analytic, required_margin
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.dynamics.truncation import (
    TruncationWarning,
    boundary_population,
    convergence_report,
    grown_caps,
    regrid_state,
)
from qutip_trap.light.raman import lamb_dicke_parameters
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics
from qutip_trap.run.job import RunError
from qutip_trap.run.levels import within_budget
from qutip_trap.run.space import (
    DETUNING_GUARD_FACTOR,
    ModeContribution,
    cap_for,
    cap_requirement,
    classify,
    drive_operator_nonzeros,
    frozen_excitation_bounds,
    select_space,
    waveform_contributions,
)
from qutip_trap.units import TWO_PI
from tests.fixtures import run
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
)
from tests.m6_fixtures import circuit_fixture
from tests.m9_fixtures import Y_MODES_TWO_IONS, tilted_pair_device

OPTS = SolverOptions()
BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))

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
    assert classify(ModeContribution(0, 1e-5, 0.03, 0.01, 0.05, 0.0), **kw) == "frozen"
    # below the drop pair: dropped whatever couples to the mode (the criterion is the contribution pair alone)
    assert (
        classify(
            ModeContribution(0, 1e-7, 1e-5, 0.01, 0.05, 0.0),
            coupled=False,
            freeze_alpha_max=1e-4,
            freeze_chi_max_rad=0.05,
        )
        == "dropped"
    )
    assert classify(ModeContribution(0, 1e-7, 1e-5, 0.01, 0.05, 0.0), **kw) == "dropped"
    assert classify(ModeContribution(0, 1e-5, 0.06, 0.01, 0.05, 0.0), **kw) == "resolved"
    assert classify(ModeContribution(0, 2e-4, 0.001, 0.01, 0.05, 0.0), **kw) == "resolved"
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
    # PLAN.md 5.1.1's table gives 6e-16 for N_exc = 10 at eta = 0.1: the operator-identity bar of 1e-12
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
    min(0, the first start), which rotates a Fock component by e^{-i n omega t0} more."""
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
    from qutip_trap.calibration.entangling import calibrate_entangling_angle, exact_gate_check, ms_schedule

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
    sched_pulses = list(ms_schedule(runs["frozen"].waveform, (0, 1), drives, table).pulses)
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


# ---- the propagator cache's fingerprint carries the device ------------------------------------------------------------------


@pytest.fixture(scope="module")
def carrier_pair():  # type: ignore[no-untyped-def]
    """A pi/2 carrier on an internal-state-only space (every mode frozen), so the propagator cache of Section 11.3 item 5 is
    live, plus a second device that differs ONLY in beam 0's wavelength: every eta changes, no mode frequency does."""
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, _stark = derived_seeds(dev, drives)
    pulse = single_qubit_pulse(
        0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
    )
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    space = HilbertSpace((2, 2), (), None, tuple(range(len(dev.crystal.modes))))
    changed = dataclasses.replace(
        dev,
        beams=(dataclasses.replace(dev.beams[0], wavelength_m=344.35e-9),) + tuple(dev.beams[1:]),
    )
    return dev, changed, sched, space


def test_the_fingerprint_carries_the_device_so_no_propagator_is_served_across_devices(carrier_pair) -> None:  # type: ignore[no-untyped-def]
    """H(t) depends on the device through every eta, the beam pointing and the geometric phase: one engine over two devices
    integrates each device's own propagator (P1[0] = 0.4950194835 and 0.4948647699)."""
    dev, changed, sched, space = carrier_pair
    pulse = sched.pulses[0]
    dk, dk2 = pulse.drive.delta_k(dev.beams), pulse.drive.delta_k(changed.beams)
    etas, _ = lamb_dicke_parameters(dev, 0, dk)
    etas2, _ = lamb_dicke_parameters(changed, 0, dk2)
    coupled = [m for m, e in etas.items() if abs(e) > 1e-12]
    assert coupled and all(abs(etas[m] - etas2[m]) > 1e-3 * abs(etas[m]) for m in coupled)
    assert [m.omega_rad_s for m in dev.crystal.modes] == [m.omega_rad_s for m in changed.crystal.modes], (
        "the fixture must change the etas alone, so that only the device can distinguish the two Hamiltonians"
    )
    fp = build_hamiltonian(dev, (pulse,), space, sample=quiet_sample()).fingerprint
    fp2 = build_hamiltonian(changed, (pulse,), space, sample=quiet_sample()).fingerprint
    assert fp and fp != fp2
    opts = SolverOptions()
    engine = JointExactEngine()
    p1: dict[str, float] = {}
    for name, device in (("base", dev), ("changed", changed)):
        tr = engine.run_pulses(
            device, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), opts
        )
        rep = engine.last_report
        assert rep is not None
        assert rep.propagator_solves == 1 and rep.propagator_cache_hits == 0, name
        p1[name] = float(np.real(np.diag(np.asarray(tr.final.internal.full())))[2])
    fresh = JointExactEngine()
    tr = fresh.run_pulses(
        changed, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), opts
    )
    p1_fresh = float(np.real(np.diag(np.asarray(tr.final.internal.full())))[2])
    assert p1["base"] == pytest.approx(0.4950194835, abs=5e-10)
    assert p1["changed"] == pytest.approx(0.4948647699, abs=5e-10)
    assert p1["changed"] == pytest.approx(p1_fresh, abs=1e-12)
    assert abs(p1["base"] - p1["changed"]) > 1e-4


# ---- the Section 11.5 guards decide on the declaration, before anything is allocated ----------------------------------------


def test_a_space_beyond_the_guards_is_measured_and_refused_without_allocating(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Section 11.5: the monitor "refuses to build" joint spaces above the guards, so declaring one allocates nothing (no
    O(D) joint identity, about 86 GB for the eight-ion eight-mode case of Section 5.4)."""
    import qutip_trap.dynamics.space as space_mod

    def refuse(self: HilbertSpace) -> qt.Qobj:
        raise AssertionError("the declaration allocated the joint identity")

    monkeypatch.setattr(space_mod.HilbertSpace, "identity", refuse)
    for n_ions, n_modes, d in ((8, 8, 16), (2, 5, 16)):
        declaration = HilbertSpace(
            (2,) * n_ions,
            tuple(ModeTruncation(m, d, (0, 4), 0.1) for m in range(n_modes)),
            None,
            (),
        )
        assert declaration.dimension == (2**n_ions) * d**n_modes
        ok, dim, nnz = within_budget(declaration, SolverOptions())
        assert not ok and dim == declaration.dimension and nnz > SolverOptions().nnz_max
    # a space inside the guards still gets the joint-identity assertion, so the monkeypatch must fire for it
    with pytest.raises(AssertionError, match="allocated the joint identity"):
        HilbertSpace((2, 2), (ModeTruncation(0, 4, (0, 1), 0.1),), None, ())


@pytest.fixture(scope="module")
def bell_schedule():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
    sched = make_schedule(compile_to_native(BELL, fx.device), fx.device, sur.table, t0_s=0.0)
    return fx, sur, sched


def test_the_selection_reports_the_guard_verdict_of_its_declaration(bell_schedule) -> None:  # type: ignore[no-untyped-def]
    """``select_space`` evaluates the Section 11.5 guards on the declaration and reports them."""
    fx, _sur, sched = bell_schedule
    inside = select_space(fx.device, sched, SolverOptions(), nbar={})
    assert inside.budget == within_budget(inside.space, SolverOptions())
    assert inside.budget.inside and not any("outside the Section 11.5 guards" in n for n in inside.notes)
    tight = SolverOptions(nnz_max=1000)
    outside = select_space(fx.device, sched, tight, nbar={})
    assert not outside.budget.inside and outside.budget.nnz > 1000
    assert any("outside the Section 11.5 guards" in n for n in outside.notes)


@pytest.mark.slow
def test_the_non_zero_guard_routes_a_run_to_gate_local(bell_schedule) -> None:  # type: ignore[no-untyped-def]
    """Section 9.17 row "Size guard": the drive-operator non-zero estimate is the other half of the guard; ``nnz_max`` below
    the estimate reroutes to GATE_LOCAL and the run says so."""
    fx, sur, _sched = bell_schedule
    joint = run(BELL, fx.device, 20, level="auto", table=sur.table)  # type: ignore[arg-type]
    assert joint.diagnostics.level == "JOINT_EXACT"
    _ok, dim, nnz = within_budget(joint.diagnostics.space, SolverOptions())
    assert nnz > dim, "the non-zero estimate N 2^N prod d_m^2 exceeds the dimension for this fixture"
    guarded = run(
        BELL,
        fx.device,
        20,
        level="auto",
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(nnz_max=nnz // 2, joint_dimension_max=10**9)),
    )
    assert guarded.diagnostics.level == "GATE_LOCAL" and guarded.diagnostics.gate_local is not None
    assert any(f"{nnz} drive non-zeros" in a for a in guarded.diagnostics.approximations)


# ---- the per-mode ceiling is configurable and its clamp is reported ----------------------------------------------------------


def test_cap_requirement_and_the_mode_dimension_ceiling() -> None:
    """Section 5.3: a Doppler-cooled nbar ~ 20 mode needs d_m >~ 150 for a boundary population below 1e-4; the ceiling
    clamps d and the declared expected range together."""
    d_want, n_hi = cap_requirement(0.0, 20.0, 0.1, d_min=6, tail=1e-4)
    assert d_want > 150 and n_hi > 140, (d_want, n_hi)
    clamped = cap_for(0.0, 20.0, 0.1, d_min=6, d_max=64, tail=1e-4)
    assert clamped.d == 64 and clamped.expected_n_range == (0, 63)
    roomy = cap_for(0.0, 20.0, 0.1, d_min=6, d_max=256, tail=1e-4)
    assert roomy.d == d_want and roomy.expected_n_range == (0, n_hi)
    assert SolverOptions().mode_dimension_max == 64
    with pytest.raises(ValueError, match="mode_dimension_max"):
        SolverOptions(mode_dimension_max=1)


def test_select_space_reads_mode_dimension_max_warns_and_names_the_clamp(bell_schedule) -> None:  # type: ignore[no-untyped-def]
    """The ceiling reaches the selection from ``SolverOptions``: a clamp warns once per clamped mode with the mode and both
    numbers, and is a note naming the range the rule asked for and the range that survives; a roomy ceiling is silent."""
    fx, _sur, sched = bell_schedule
    nbar = {m: 20.0 for m in range(len(fx.device.crystal.modes))}
    with pytest.warns(TruncationWarning) as caught:
        tight = select_space(fx.device, sched, SolverOptions(mode_dimension_max=8), nbar=nbar)
    assert tight.resolved_modes, "the fixture must resolve at least one mode for the clamp to bite"
    messages = [str(w.message) for w in caught if issubclass(w.category, TruncationWarning)]
    assert len(messages) == len(tight.resolved_modes), messages
    assert any(
        re.search(r"mode 2: the cap rule asks for d = \d+ .* mode_dimension_max = 8 clamps it to d = 8", m)
        for m in messages
    ), messages
    assert all(tight.space.truncation(m).d == 8 for m in tight.resolved_modes)
    clamp = [n for n in tight.notes if "mode_dimension_max = 8 clamps it" in n]
    assert len(clamp) == len(tight.resolved_modes), tight.notes
    assert "the cap rule asks for d = " in clamp[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        roomy = select_space(fx.device, sched, SolverOptions(mode_dimension_max=512), nbar=nbar)
    assert all(roomy.space.truncation(m).d > 8 for m in roomy.resolved_modes)
    assert not any("clamps it" in n for n in roomy.notes)


# ---- Section 9.9's convergence regime -----------------------------------------------------------------------------------------


def test_convergence_report_runs_all_three_arms_of_section_9_9() -> None:
    """Tolerances / 10, tolerances x 10 (the integrator-ladder half) and every resolved cap + 2, each a ``ConvergenceReport``;
    the cap arm respects Section 9.9's dimension ceiling."""
    space = HilbertSpace(
        (2,), (ModeTruncation(0, 6, (0, 2), 0.1), ModeTruncation(1, 6, (0, 2), 0.1)), None, ()
    )
    seen: list[tuple[float, tuple[int, ...]]] = []

    def probe(opts: SolverOptions, sp: HilbertSpace) -> dict[str, np.ndarray]:
        seen.append((opts.atol, tuple(t.d for t in sp.resolved)))
        return {"p": np.array([0.5 + opts.atol, 0.5 - opts.atol])}

    rep = convergence_report(probe, SolverOptions(), space, tol=1e-6)
    atols = [a for a, _d in seen]
    dims = [d for _a, d in seen]
    # tightened arm: (1e-10, 1e-11); loosened arm: (1e-9, 1e-10); cap arm: 1e-10 on the base and on the grown space
    assert atols == pytest.approx([1e-10, 1e-11, 1e-9, 1e-10, 1e-10, 1e-10])
    assert dims == [(6, 6)] * 5 + [(8, 8)]
    assert rep.grown_modes == (0, 1) and rep.add == 2
    assert rep.tightened.tolerances == (1e-10, 1e-8)
    assert rep.tightened.tightened_tolerances == pytest.approx((1e-11, 1e-9))
    assert rep.loosened.tolerances == pytest.approx((1e-9, 1e-7))
    assert rep.tightened.max_change == pytest.approx(9e-11)
    assert rep.loosened.max_change == pytest.approx(9e-10)
    # the cap arm varies the space at UNCHANGED tolerances, so its two tolerance pairs are equal
    assert rep.caps.tolerances == rep.caps.tightened_tolerances == (1e-10, 1e-8)
    assert rep.caps.max_change == 0.0
    assert rep.converged and "converged" in rep.summary()
    # the ceiling: 2 x 6 x 6 = 72, and +2 on both modes would reach 2 x 8 x 8 = 128
    capped = grown_caps(space, 2, dimension_max=100)
    assert [t.d for t in capped.resolved] == [8, 6], "the second mode crosses the ceiling and is left"
    assert grown_caps(space, 2).dimension == 128


def test_convergence_regime_of_the_bell_circuit() -> None:
    """Section 9.9 on the Section 9.6 Bell circuit: tolerances tightened and loosened by ten and every resolved cap raised by
    two move the probabilities by less than the test tolerance; the cap arm regrids the prepared state (``regrid_state``)."""
    dev = two_ion_device()
    drives = raman_gate_drives(2)
    modes = gate_modes(dev, (0, 1), (0, 1))
    wf = symmetric_pulse(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3, pair=(0, 1)).waveform
    chi = abs(waveform_integrals(wf, modes).chi_of(0, 1))
    assert chi > 0.1
    pulses = entangling_pulses(
        wf,
        drives,
        spin_phases_rad={0: -math.pi / 2, 1: -math.pi / 2},
        t_start_s=0.0,
        table=table_with_waveform((0, 1), wf),
        gate_id="ms",
    )
    sched = Schedule(tuple(pulses), (), (), {0: 0.0, 1: 0.0})
    base = HilbertSpace(
        (2, 2),
        tuple(ModeTruncation(m, 8, (0, 3), 0.13) for m in (2, 3)),
        None,
        tuple(m for m in range(len(dev.crystal.modes)) if m not in (2, 3)),
    )
    state0 = base.initial_state([0, 0])
    assert state0.joint is not None

    def probe(opts: SolverOptions, sp: HilbertSpace) -> dict[str, np.ndarray]:
        st = state0
        if sp is not base:
            st = dataclasses.replace(state0, joint=regrid_state(state0.joint, base, sp))
        tr = JointExactEngine().run_pulses(dev, sched, st, sp, quiet_sample(), SeedSpec(0), opts)
        return {"register_populations": np.real(np.diag(np.asarray(tr.final.internal.full())))}

    rep = convergence_report(probe, SolverOptions(), base, tol=1e-4)
    assert rep.converged, rep.summary()
    assert rep.grown_modes == (2, 3)


@pytest.mark.slow
def test_convergence_regime_of_the_frozen_spectator_fixture() -> None:
    """Section 9.9 on the tilted Raman pair (y-COM 120 kHz below the tone, eta_y = 0.008): a run whose space carries a
    genuine frozen spectator, built by hand as the frozen-spectator comparison above builds it."""
    dev = tilted_pair_device(math.radians(6.0))
    drives = raman_gate_drives(2)
    modes = gate_modes(dev, (0, 1), (0, 1))
    com = modes.modes[-1]
    wf = symmetric_pulse(modes, gate_mode=com, loops=1, epsilon_hz=20e3, pair=(0, 1)).waveform
    pulses = entangling_pulses(
        wf,
        drives,
        spin_phases_rad={0: -math.pi / 2, 1: -math.pi / 2},
        t_start_s=0.0,
        table=table_with_waveform((0, 1), wf),
        gate_id="ms",
    )
    sched = Schedule(tuple(pulses), (), (), {0: 0.0, 1: 0.0})
    y_rock, y_com = Y_MODES_TWO_IONS
    space = HilbertSpace(
        (2, 2),
        (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13)),
        None,
        (0, 1, y_rock, y_com),
    )
    assert space.resolved and space.frozen

    def probe(opts: SolverOptions, sp: HilbertSpace) -> dict[str, np.ndarray]:
        tr = JointExactEngine().run_pulses(
            dev, sched, sp.initial_state([0, 0]), sp, quiet_sample(), SeedSpec(0), opts
        )
        return {"register_populations": np.real(np.diag(np.asarray(tr.final.internal.full())))}

    rep = convergence_report(probe, SolverOptions(margin_check=False), space, tol=1e-4)
    assert rep.converged, rep.summary()
    assert rep.grown_modes == (2, 3)


@pytest.mark.slow
def test_run_reports_the_section_5_5_tolerance_convergence_when_asked(bell_schedule) -> None:  # type: ignore[no-untyped-def]
    """Section 5.5's second bullet through the run: ``convergence_check`` repeats the evolution at atol and rtol tightened by
    ten and puts the change in the register populations on ``Diagnostics.convergence``; None means not asked for."""
    fx, sur, _sched = bell_schedule
    kw = dict(table=sur.table, seed=5)
    plain = run(BELL, fx.device, 20, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    assert plain.diagnostics.convergence is None
    checked = run(
        BELL,
        fx.device,
        20,
        level="JOINT_EXACT",
        **kw,
        numerics=Numerics.from_solver_options(SolverOptions(convergence_check=True)),
    )
    rep = checked.diagnostics.convergence
    assert rep is not None
    assert rep.tolerances == (1e-10, 1e-8)
    assert rep.tightened_tolerances == pytest.approx((1e-11, 1e-9))
    assert set(rep.changes) == {"register_populations"}
    assert rep.converged, rep.summary()
    assert any("tolerance convergence" in a for a in checked.diagnostics.approximations)


# ---- the ENR option end to end --------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_an_enr_group_evolves_as_one_factor_and_run_refuses_the_hot_group_within_the_guards(
    bell_schedule,
) -> None:  # type: ignore[no-untyped-def]
    """Section 11.3 item 1. On the engine: the played Bell schedule on two spaces that differ only in how the two y modes
    (eta = 0 for a Delta k along x) are carried, two resolved factors of dimension 3 (3960) or ONE ENR factor at N_exc = 2
    (2640): the register agrees to the integration tolerance, the class map reports ``enr`` and the boundary report covers
    the top ENR shell. Through the run: this fixture's y modes are Doppler-limited (nbar 3.9 and 3.6), not the cold undriven
    group the ENR option is for, so N_exc = 10 (a 37752-dimensional declaration) is refused inside the Section 11.5 guards
    and N_exc = 2 trips the top shell, whose growth to 16016 dimensions the engine refuses."""
    fx, sur, sched = bell_schedule
    y_rock, y_com = Y_MODES_TWO_IONS
    x_caps = (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13))
    product = HilbertSpace(
        (2, 2),
        x_caps + (ModeTruncation(y_rock, 3, (0, 0), 0.0), ModeTruncation(y_com, 3, (0, 0), 0.0)),
        None,
        (0, 1),
    )
    enr = HilbertSpace((2, 2), x_caps, ((y_rock, y_com), 2), (0, 1))
    assert product.dimension == 3960 and enr.dimension == 2640 and enr.dims == [2, 2, 10, 11, 6]
    assert (
        enr.mode_class(y_rock) == "enr"
        and enr.mode_class(y_com) == "enr"
        and enr.enr_group == ((y_rock, y_com), 2)
    )
    opts = SolverOptions(
        margin_check=False
    )  # the x caps are the frozen-spectator test's; the comparison is the subject
    finals = {}
    for name, sp in (("product", product), ("enr", enr)):
        eng = JointExactEngine()
        tr = eng.run_pulses(fx.device, sched, sp.initial_state([0, 0]), sp, quiet_sample(), SeedSpec(0), opts)
        finals[name] = np.real(np.diag(np.asarray(tr.final.internal.full())))
        rep = eng.last_report
        assert rep is not None
        if name == "enr":
            reported = set().union(*(set(seg.boundary_population) for seg in rep.segments if seg.pulses))
            assert {y_rock, y_com} <= reported, (
                reported
            )  # the top ENR shell is the group's boundary (Section 5.1)
    # the same physics on different dimensions: the difference is the integrator's tolerance over the schedule, inside the
    # ~5e-7 in norm that Section 11.1 calls identical
    assert np.max(np.abs(finals["enr"] - finals["product"])) < 1e-7, finals
    assert 0.4 < finals["enr"][0] < 0.6, finals["enr"]  # a Bell state's register populations
    kw = dict(table=sur.table, keep_final_state=True, seed=7)
    with pytest.raises(RunError, match="37752"):
        run(
            BELL,
            fx.device,
            200,
            level="JOINT_EXACT",
            **kw,
            numerics=Numerics.from_solver_options(enr_group=((y_rock, y_com), 10)),
        )  # type: ignore[arg-type]
    with pytest.raises((TruncationLimit, RunError), match="16016"):
        run(
            BELL,
            fx.device,
            200,
            level="JOINT_EXACT",
            **kw,
            numerics=Numerics.from_solver_options(enr_group=((y_rock, y_com), 2)),
        )  # type: ignore[arg-type]


def test_every_joint_operator_and_state_of_an_enr_space_carries_the_spaces_dims() -> None:
    """Section 5.1.1: the ENR group is ONE tensor factor of dimension C(M + N_exc, N_exc) in ``HilbertSpace.dims``, and every
    joint object the space hands out carries those dims (QuTiP's ``enr_*`` constructors label the group by its per-mode
    dims), so a QobjEvo of motional terms plus a factorized drive term builds."""
    space = HilbertSpace(
        (2, 2), (ModeTruncation(2, 5, (0, 2), 0.1), ModeTruncation(3, 4, (0, 1), 0.1)), ((4, 5), 2), (0, 1)
    )
    assert space.dims == [2, 2, 5, 4, 6] and space.dimension == 480
    want = [space.dims, space.dims]
    for op in (
        space.identity(),
        space.number(4),
        space.annihilation(5),
        space.number(2),
        space.sigma_plus(0),
    ):
        assert op.dims == want, op.dims
        assert op.shape == (480, 480)
    ket = space.initial_state([0, 0]).joint
    assert ket is not None and ket.dims[0] == space.dims and ket.shape == (480, 1), ket.dims
    dm = space.product_state(
        [0, 0],
        {2: space.fock(2, 0), 3: space.fock(3, 0), -1: space.enr_state(thermal={4: 0.1, 5: 0.2})},
    )
    assert dm.dims == want and abs(dm.tr() - 1.0) < 1e-12
    drive = space.drive_operator_factorized(0, {2: 0.1, 3: 0.05})
    h = qt.QobjEvo([space.number(2) * 1e6, space.number(4) * 2e6, [drive, qt.coefficient(_cos_2pi)]])
    assert h.dims == want and h(0.0).shape == (480, 480)
    # the ENR factor's own state is exactly the relabelled tensor factor: index sums, not ptrace, give its marginal
    assert space.enr_state(fock={4: 1, 5: 0}).shape == (6, 1)


def test_a_small_enr_cap_trips_the_monitor_and_grown_enr_recovers() -> None:
    """Section 5.5 on the top ENR shell: a cap so small that the displaced state reaches it trips the boundary monitor, and
    ``grown_enr`` plus ``regrid_state`` carry the state onto the larger group."""
    small = HilbertSpace((2,), (), ((0, 1), 2), ())
    assert small.dims == [2, 6]
    st = small.initial_state([0], fock={0: 0, 1: 0})
    assert st.joint is not None
    displaced = small.embed(small.enr_displacement({0: 0.8, 1: 0.8}), 1) * st.joint
    displaced = displaced / displaced.norm()
    top = boundary_population(displaced, small, 0)
    assert top > 1e-6, top
    big = small.grown_enr(6)
    assert big.enr_group == ((0, 1), 8) and big.dims == [2, 45]
    carried = regrid_state(displaced, small, big)
    assert abs(carried.norm() - 1.0) < 1e-12
    assert boundary_population(carried, big, 0) < top


# ---- the integrator ladder --------------------------------------------------------------------------------------------------


def _tiny_ladder_problem() -> tuple[qt.QobjEvo, qt.Qobj, np.ndarray]:
    H = qt.QobjEvo([qt.sigmaz() * TWO_PI * 1e5, [qt.sigmax() * TWO_PI * 1e5, qt.coefficient(_cos_2pi)]])
    return H, qt.basis(2, 0), np.linspace(0.0, 1e-5, 3)


def _cos_2pi(t: float) -> float:
    return float(np.cos(TWO_PI * 1e5 * t))


@pytest.mark.parametrize("d_m", [121, 151, 201])
def test_the_atol_keying_of_section_5_3_at_the_plans_validated_points(d_m: int) -> None:
    """Section 5.3 keys atol to the measured points, 1e-10 up to d_m ~ 100 and 1e-8 above, validated at d_m = 121, 151, 201."""
    H, psi0, times = _tiny_ladder_problem()
    assert d_m > LARGE_MODE_DIMENSION
    ev = evolve(H, psi0, times, largest_mode_dimension=d_m)
    assert ev.atol == pytest.approx(1e-8) and ev.retries == ()
    below = evolve(H, psi0, times, largest_mode_dimension=LARGE_MODE_DIMENSION)
    assert below.atol == pytest.approx(1e-10)


def test_a_programming_error_is_not_recorded_as_an_integrator_failure() -> None:
    """A mismatched ``e_op`` or an unsupported solver option propagates as itself, never as an integrator failure retried on
    every rung, and leaves no retry behind."""
    H, psi0, times = _tiny_ladder_problem()
    with pytest.raises(ValueError, match="incompatible dimensions"):
        evolve(H, psi0, times, e_ops={"x": qt.qeye(3)})
    # 'diag' takes no atol/rtol/nsteps and raises KeyError before it integrates anything
    with pytest.raises(KeyError, match="not supported"):
        evolve(H, psi0, times, options=SolverOptions(integrators=("diag", "dop853")))


def _two_tone(t: float, Om: float, mu: float, tag: tuple[int, int] = (0, 0)) -> float:
    return Om * np.cos(mu * t)


def _ms_hamiltonian(nmodes: int, d_m: int) -> qt.QobjEvo:
    """The two-ion bichromatic MS Hamiltonian of the Section 11.1 timing fixture: CSR drive operators, one coefficient
    object per term (the tag keeps QobjEvo from merging them)."""
    omega = [TWO_PI * 3.0e6, TWO_PI * 2.8284e6, TWO_PI * 2.9e6][:nmodes]
    eta = [(0.080, 0.080), (0.0824, -0.0824), (0.047, 0.047)][:nmodes]
    eps = TWO_PI * 10e3
    a = qt.destroy(d_m)
    ions, modes = [qt.qeye(2)] * 2, [qt.qeye(d_m)] * nmodes
    h0 = sum(
        w * qt.tensor(*ions, *[a.dag() * a if k == m else modes[k] for k in range(nmodes)])
        for m, w in enumerate(omega)
    )
    terms: list[object] = [h0.to("CSR")]
    args = {"Om": eps / (2 * eta[0][0]), "mu": omega[0] - eps}
    for i in range(2):
        spin = [qt.sigmap() if k == i else qt.qeye(2) for k in range(2)]
        v = qt.tensor(*spin, *[(1j * eta[m][i] * (a + a.dag())).expm() for m in range(nmodes)]).to("CSR")
        for j, op in enumerate((v, v.dag())):
            terms.append([op, qt.coefficient(_two_tone, args={**args, "tag": (i, j)})])
    return qt.QobjEvo(terms)


@pytest.mark.slow
def test_the_ladder_escalates_when_a_rung_fails_and_records_the_retry() -> None:
    """A failing rung is recorded and the next rung carries the integration. The trigger is the step budget (the plan's
    stiffness trigger does not reproduce: ``conv.integrator_ladder_stiffness``): on the Section 11.1 row at dimension 256 the
    fifth-order ``tsit5`` needs more than 2000 steps per output interval and succeeds at 4000, the eighth-order ``dop853``
    more than 500 and succeeds at 1000, so ``("tsit5", "dop853")`` at nsteps = 2000 aborts on its first rung and finishes on
    its second."""
    nmodes, d_m = 2, 8
    H = _ms_hamiltonian(nmodes, d_m)
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), *[qt.basis(d_m, 0)] * nmodes)
    times = np.linspace(0.0, 20e-6, 3)
    opts = SolverOptions(nsteps=2000, integrators=("tsit5", "dop853"))
    ev = evolve(H, psi0, times, options=opts, omega_max_rad_s=TWO_PI * 3.0e6)
    assert ev.integrator == "dop853", ev.retries
    assert len(ev.retries) == 1, ev.retries
    assert ev.retries[0].startswith("tsit5@atol=1e-10")
    assert "IntegratorException" in ev.retries[0]
    assert abs(ev.final.norm() - 1.0) < 1e-6
    # the same ladder with a budget both rungs meet needs no retry at all
    roomy = evolve(
        H,
        psi0,
        times,
        options=SolverOptions(nsteps=10**7, integrators=("tsit5", "dop853")),
        omega_max_rad_s=TWO_PI * 3.0e6,
    )
    assert roomy.integrator == "tsit5" and roomy.retries == ()
    # the escalated and the un-escalated runs agree to what Section 11.1 calls identical (4.41e-7 here in norm)
    assert (roomy.final - ev.final).norm() < 1e-6, (roomy.final - ev.final).norm()


# ---- Section 9.9: mcsolve with and without improved_sampling against the mesolve histogram ----------------------------------


@pytest.mark.slow
def test_mcsolve_with_and_without_improved_sampling_converge_to_the_mesolve_histogram() -> None:
    """The last clause of Section 9.9 on a two-ion dissipative case: one resolved mode at d_m = 12 with a heating channel and
    ONE carrier pulse (dimension 48, one trajectory segment, the condition for improved sampling): both trajectory ensembles
    land on the density matrix's register populations inside their multinomial band, both mixtures have unit weight, and the
    improved-sampling ensemble carries one member more than ``ntraj``, the deterministic no-jump trajectory."""
    dev = chain_device(2)
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    drives = raman_gate_drives(2)
    rabi, _stark = derived_seeds(noisy, drives)
    pulse = single_qubit_pulse(
        0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
    )
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    space = HilbertSpace(
        (2, 2),
        (ModeTruncation(X_COM_TWO_IONS, 12, (0, 4), 0.13),),
        None,
        tuple(m for m in range(len(dev.crystal.modes)) if m != X_COM_TWO_IONS),
    )
    assert space.dimension == 48
    state = space.initial_state([0, 0])
    ntraj = 100
    base = dict(margin_check=False, map="serial")
    ref = JointExactEngine(device_channels=True).run_pulses(
        noisy,
        sched,
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(lindblad_method="mesolve", **base),  # type: ignore[arg-type]
    )
    p_ref = np.real(np.diag(np.asarray(ref.final.internal.full())))
    for improved in (True, False):
        eng = JointExactEngine(device_channels=True)
        tr = eng.run_pulses(
            noisy,
            sched,
            state,
            space,
            quiet_sample(),
            SeedSpec(11),
            SolverOptions(
                lindblad_method="mcsolve",
                ntraj=ntraj,
                improved_sampling=improved,
                **base,  # type: ignore[arg-type]
            ),
        )
        rep = eng.last_report
        assert rep is not None and rep.method == "mcsolve"
        p_mc = np.real(np.diag(np.asarray(tr.final.internal.full())))
        assert abs(float(np.sum(p_mc)) - 1.0) < 1e-9, "the mixture's weights must sum to one"
        # the multinomial band of ntraj trajectories, four sigma
        sigma = np.sqrt(np.maximum(p_ref * (1.0 - p_ref), 1e-12) / ntraj)
        assert np.all(np.abs(p_mc - p_ref) < 4.0 * sigma + 1e-3), (improved, p_mc, p_ref, sigma)
        # the no-jump member is a deterministic EXTRA member, so the stored ensemble is one larger than ntraj
        assert rep.trajectories == ntraj + (1 if improved else 0), (improved, rep.trajectories)
