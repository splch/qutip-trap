"""Regressions for the M9a/M9b audit (PLAN.md Sections 5.1, 5.1.1, 5.2, 5.3, 5.5, 9.9, 9.17, 11.1, 11.5).

Every test here pins a defect the M9 audit found:

- the device in ``BuiltHamiltonian.fingerprint``, so the engine's propagator cache cannot serve one device's propagator for
  another (audit B1);
- the Section 11.5 guards evaluated on the DECLARED space before any operator is allocated, and the non-zero branch of the
  guard exercised through ``run()`` (audit B2, and the missing 9.17 row);
- ``SolverOptions.mode_dimension_max``: the per-mode ceiling is configurable and a clamp is reported instead of silently
  narrowing the declared occupation range (audit D1/B5);
- Section 9.9's convergence regime (``hilbert.truncation.convergence_report``) on the cases Section 9.9 names (audit E6);
- the ENR option end to end through ``run(enr_group=...)`` (audit D5);
- the integrator ladder: the atol keying of Section 5.3 at the plan's three validated points, an escalation that really
  happens, and a programming error that no longer masquerades as an integrator failure (audit B8, E11).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.api import (
    Circuit,
    HilbertSpace,
    ModeTruncation,
    Operation,
    SeedSpec,
    SolverOptions,
    run,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import compile_to_native
from qutip_trap.control.schedule import Schedule, single_qubit_pulse
from qutip_trap.control.schedule import schedule as make_schedule
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.evolve import LARGE_MODE_DIMENSION, evolve
from qutip_trap.dynamics.hamiltonian import build_hamiltonian
from qutip_trap.hilbert.truncation import TruncationWarning, convergence_report, grown_caps, regrid_state
from qutip_trap.light.raman import lamb_dicke_parameters
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.options import Numerics
from qutip_trap.run.job import RunError
from qutip_trap.run.levels import within_budget
from qutip_trap.run.space import cap_for, cap_requirement, select_space
from qutip_trap.units import TWO_PI
from tests.m4_fixtures import chain_device, derived_seeds, raman_gate_drives
from tests.m6_fixtures import circuit_fixture
from tests.m9_fixtures import Y_MODES_TWO_IONS, tilted_pair_device

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


# ---- B1: the device belongs in the built Hamiltonian's fingerprint ----------------------------------------------------------


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
    """Audit B1: H(t) depends on the device through every eta, the beam pointing and the geometric phase, none of which the
    rest of the digest carries. One engine over two devices used to serve the first device's propagator for the second and
    report P1[0] = 0.4950194835 where the correct answer is 0.4948647699 (error 1.5e-4, three orders above the tolerance)."""
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
        assert rep.propagator_solves == 1 and rep.propagator_cache_hits == 0, (
            name,
            "the second device must integrate its own propagator, never hit the first device's",
        )
        p1[name] = float(np.real(np.diag(np.asarray(tr.final.internal.full())))[2])
    fresh = JointExactEngine()
    tr = fresh.run_pulses(
        changed, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), opts
    )
    p1_fresh = float(np.real(np.diag(np.asarray(tr.final.internal.full())))[2])
    # the audit's measured numbers, to their printed digits
    assert p1["base"] == pytest.approx(0.4950194835, abs=5e-10)
    assert p1["changed"] == pytest.approx(0.4948647699, abs=5e-10)
    assert p1["changed"] == pytest.approx(p1_fresh, abs=1e-12)
    assert abs(p1["base"] - p1["changed"]) > 1e-4


# ---- B2: the Section 11.5 guards decide before anything is allocated --------------------------------------------------------


def test_a_space_beyond_the_guards_is_measured_and_refused_without_allocating(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Section 11.5 says the monitor "refuses to build" joint spaces above the guards. ``HilbertSpace.check()`` used to form
    ``qt.tensor(*factor_identities())`` first, an O(D) CSR identity (measured +64 MB at D = 4.2e6 and about 86 GB for the
    eight-ion eight-mode case of Section 5.4), so the guard could not refuse what it had already built (audit B2)."""
    import qutip_trap.hilbert.space as space_mod

    def refuse(self: HilbertSpace) -> qt.Qobj:
        raise AssertionError("the declaration allocated the joint identity")

    monkeypatch.setattr(space_mod.HilbertSpace, "identity", refuse)
    # the case Section 5.4 cites: eight ions with eight modes; and the audit's measured 2 ions x 5 modes at d = 16
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


def test_the_selection_reports_the_guard_verdict_of_its_declaration() -> None:
    """``select_space`` evaluates the Section 11.5 guards on the declaration and reports them, so ``run()`` reads one verdict
    instead of recomputing it after the space is in hand (audit B2)."""
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
    sched = make_schedule(compile_to_native(BELL, fx.device), fx.device, sur.table, t0_s=0.0)
    inside = select_space(fx.device, sched, SolverOptions(), nbar={})
    assert inside.budget == within_budget(inside.space, SolverOptions())
    assert inside.budget[0] and not any("outside the Section 11.5 guards" in n for n in inside.notes)
    tight = SolverOptions(nnz_max=1000)
    outside = select_space(fx.device, sched, tight, nbar={})
    assert not outside.budget[0] and outside.budget[2] > 1000
    assert any("outside the Section 11.5 guards" in n for n in outside.notes)


@pytest.mark.slow
def test_the_non_zero_guard_routes_a_run_to_gate_local() -> None:
    """Section 9.17 row "Size guard": the drive-operator non-zero estimate is the OTHER half of the guard, and no test ever
    drove a run through it (only ``joint_dimension_max``). ``nnz_max`` below the estimate reroutes to GATE_LOCAL and the run
    says so (audit E2)."""
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
    kw = dict(
        table=sur.table,
    )
    joint = run(BELL, fx.device, 20, level="auto", **kw)  # type: ignore[arg-type]
    assert joint.diagnostics.level == "JOINT_EXACT"
    _ok, dim, nnz = within_budget(joint.diagnostics.space, SolverOptions())
    assert nnz > dim, "the non-zero estimate N 2^N prod d_m^2 exceeds the dimension for this fixture"
    guarded = run(
        BELL,
        fx.device,
        20,
        level="auto",
        **kw,
        numerics=Numerics.from_solver_options(SolverOptions(nnz_max=nnz // 2, joint_dimension_max=10**9)),
    )
    assert guarded.diagnostics.level == "GATE_LOCAL" and guarded.diagnostics.gate_local is not None
    assert any(f"{nnz} drive non-zeros" in a for a in guarded.diagnostics.approximations)


# ---- D1/B5: the per-mode ceiling is configurable and its clamp is reported ---------------------------------------------------


def test_cap_requirement_and_the_mode_dimension_ceiling() -> None:
    """Section 5.3: a Doppler-cooled nbar ~ 20 mode needs d_m >~ 150 for a boundary population below 1e-4. ``cap_for``'s
    ceiling used to be a hard-coded 64 that silently narrowed the declared expected range with it (audit D1)."""
    d_want, n_hi = cap_requirement(0.0, 20.0, 0.1, d_min=6, tail=1e-4)
    assert d_want > 150 and n_hi > 140, (d_want, n_hi)
    clamped = cap_for(0.0, 20.0, 0.1, d_min=6, d_max=64, tail=1e-4)
    assert clamped.d == 64 and clamped.expected_n_range == (0, 63)
    roomy = cap_for(0.0, 20.0, 0.1, d_min=6, d_max=256, tail=1e-4)
    assert roomy.d == d_want and roomy.expected_n_range == (0, n_hi)
    assert SolverOptions().mode_dimension_max == 64
    with pytest.raises(ValueError, match="mode_dimension_max"):
        SolverOptions(mode_dimension_max=1)


def test_select_space_reads_mode_dimension_max_and_names_the_clamp() -> None:
    """The ceiling reaches the selection from ``SolverOptions`` and a clamp is a note that names the mode, the range the rule
    asked for and the range that survives (audit E9: it was silent)."""
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
    sched = make_schedule(compile_to_native(BELL, fx.device), fx.device, sur.table, t0_s=0.0)
    nbar = {m: 20.0 for m in range(len(fx.device.crystal.modes))}
    with pytest.warns(TruncationWarning) as caught:  # 0.2.0: the clamp is also said out loud (api plan 1.9)
        tight = select_space(fx.device, sched, SolverOptions(mode_dimension_max=8), nbar=nbar)
    assert tight.resolved_modes, "the fixture must resolve at least one mode for the clamp to bite"
    assert len([w for w in caught if issubclass(w.category, TruncationWarning)]) == len(tight.resolved_modes)
    assert all(tight.space.truncation(m).d == 8 for m in tight.resolved_modes)
    clamp = [n for n in tight.notes if "mode_dimension_max = 8 clamps it" in n]
    assert len(clamp) == len(tight.resolved_modes), tight.notes
    assert "the cap rule asks for d = " in clamp[0]
    roomy = select_space(fx.device, sched, SolverOptions(mode_dimension_max=512), nbar=nbar)
    assert all(roomy.space.truncation(m).d > 8 for m in roomy.resolved_modes)
    assert not any("clamps it" in n for n in roomy.notes)


# ---- E6: Section 9.9's convergence regime -----------------------------------------------------------------------------------


def test_convergence_report_runs_all_three_arms_of_section_9_9() -> None:
    """The three comparisons are tolerances / 10, tolerances x 10 (the integrator-ladder half) and every resolved cap + 2,
    each one a ``dynamics.evolve.ConvergenceReport`` (M2's type, reused rather than reinvented), and the cap arm respects
    Section 9.9's dimension ceiling ("where the doubled cap stays within the dimension ceiling; otherwise the check runs on
    the reduced mode set")."""
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


def test_convergence_regime_of_the_bell_circuit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Section 9.9 on the Section 9.6 Bell circuit: tightening and loosening the tolerances by ten and raising every resolved
    cap by two move the reported probabilities by less than the stated test tolerance. The cap arm regrids the prepared state
    onto the grown space (``regrid_state``), which is what makes the two runs comparable."""
    from qutip_trap.control.schedule import entangling_pulses
    from qutip_trap.control.shaping import gate_modes, symmetric_pulse, waveform_integrals
    from tests.m4_fixtures import X_COM_TWO_IONS, table_with_waveform, two_ion_device

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
    """Section 9.9 on the frozen-spectator fixture of ``tests/m9_fixtures.py`` (the tilted Raman pair, y-COM 120 kHz below the
    tone, eta_y = 0.008): the same three comparisons on a run whose space carries a genuine frozen spectator.

    The space is built by hand exactly as ``tests/test_scaling_modes.py``'s frozen-spectator comparison builds it, not through
    ``select_space``: the criterion reads a schedule's PLAYED entangling gates, and a hand-assembled ``Schedule`` carries
    pulses without the ``PlayedGate`` records, so ``select_space`` would see no entangling gate and resolve nothing."""
    from qutip_trap.control.schedule import entangling_pulses
    from qutip_trap.control.shaping import gate_modes, symmetric_pulse
    from tests.m4_fixtures import table_with_waveform

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


# ---- D5: the ENR option end to end ------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_an_enr_group_evolves_as_one_factor_and_run_refuses_the_hot_group_within_the_guards() -> None:
    """Section 11.3 item 1 / audit D5: nothing in the tree ever evolved an ENR group through the engine, and nothing ever
    passed ``enr_group=`` to ``run()``.

    Positive half, on the engine: the played Bell schedule of the Section 9.6 two-ion fixture on two hand-built spaces that
    agree everywhere but in how the two y modes are carried - as two resolved factors of dimension 3 ([2, 2, 10, 11, 3, 3],
    3960) or as ONE excitation-number-restricted factor at N_exc = 2 ([2, 2, 10, 11, 6], 2640) - both starting with the y
    modes in their ground state. A Delta k along x gives those modes eta = 0 exactly, so the ENR factor evolves under H_0
    alone and the register must agree to the integration tolerance (1e-8, measured 2.4e-9); the class map reports ``enr`` for both members, the space carries the group,
    and the engine's boundary report covers the top ENR shell.

    Negative half, through ``run()``: this fixture's y modes are Doppler-limited (prepared nbar 3.9 and 3.6; the 355 nm pair
    carries no k along y to sideband-cool them), so they are NOT the "cold undriven group" Section 5.1.1's ENR option is for.
    Asking for them anyway is refused inside the Section 11.5 guards: N_exc = 10 declares a 37752-dimensional joint space
    (the first version of this test built it, took 20 GB and an hour, and then failed inside QobjEvo on the dims mismatch
    fixed by ``HilbertSpace.with_space_dims``), and N_exc = 2 trips the top shell, whose growth to N_exc = 6 would be a
    16016-dimensional space, so the engine refuses the growth instead of building it."""
    from qutip_trap.dynamics.engine import TruncationLimit

    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
    sched = make_schedule(compile_to_native(BELL, fx.device), fx.device, sur.table, t0_s=0.0)
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
    # the two constructions integrate the same physics on different dimensions: the difference is the integrator's
    # tolerance over the played schedule (measured 2.4e-9 at atol 1e-10, rtol 1e-8 in the Schroedinger picture, 2.9e-8 in
    # the rotating frame the engine integrates in since 2026-09-09, both inside the ~5e-7 in norm that Section 11.1 calls
    # identical), not the ENR construction, whose operator identity is pinned to 1e-12 in tests/test_scaling_modes.py
    assert np.max(np.abs(finals["enr"] - finals["product"])) < 1e-7, finals
    assert 0.4 < finals["enr"][0] < 0.6, finals["enr"]  # a Bell state's register populations

    kw = dict(
        table=sur.table,
        keep_final_state=True,
        seed=7,
    )
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
    """Section 5.1.1: the ENR group is ONE tensor factor of dimension C(M + N_exc, N_exc) in ``HilbertSpace.dims``, and the
    factorized drive kernel builds its operators on those dims. QuTiP's ``enr_*`` constructors label the group by its per-mode
    dims (n_exc + 1 each) instead, so until 2026-09-08 ``identity``, ``number``, ``annihilation``, ``embed`` and the joint
    states came back labelled [..., 3, 3] while the drive term said [..., 6], and ``QobjEvo`` refused to combine them: the
    first ENR run through ``run()`` failed on exactly that after building a 37752-dimensional space. Every joint object the
    space hands out now carries ``space.dims``, and a QobjEvo of motional terms plus a factorized drive term builds."""
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
    ``grown_enr`` plus ``regrid_state`` carry the state onto the larger group (audit D5)."""
    from qutip_trap.hilbert.truncation import boundary_population

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


# ---- B8/E11: the integrator ladder ------------------------------------------------------------------------------------------


def _tiny_ladder_problem() -> tuple[qt.QobjEvo, qt.Qobj, np.ndarray]:
    H = qt.QobjEvo([qt.sigmaz() * TWO_PI * 1e5, [qt.sigmax() * TWO_PI * 1e5, qt.coefficient(_cos_2pi)]])
    return H, qt.basis(2, 0), np.linspace(0.0, 1e-5, 3)


def _cos_2pi(t: float) -> float:
    return float(np.cos(TWO_PI * 1e5 * t))


@pytest.mark.parametrize("d_m", [121, 151, 201])
def test_the_atol_keying_of_section_5_3_at_the_plans_validated_points(d_m: int) -> None:
    """Section 5.3 keys atol to the measured points, 1e-10 up to d_m ~ 100 and 1e-8 above, "validated at d_m = 121, 151 and
    201". Nothing pinned the keying; this does, at all three."""
    H, psi0, times = _tiny_ladder_problem()
    assert d_m > LARGE_MODE_DIMENSION
    ev = evolve(H, psi0, times, largest_mode_dimension=d_m)
    assert ev.atol == pytest.approx(1e-8) and ev.retries == ()
    below = evolve(H, psi0, times, largest_mode_dimension=LARGE_MODE_DIMENSION)
    assert below.atol == pytest.approx(1e-10)


def test_a_programming_error_is_not_recorded_as_an_integrator_failure() -> None:
    """Audit B8: the ladder caught bare ``Exception``, so a mismatched ``e_op`` or an unsupported solver option was recorded as
    an integrator failure, retried on every remaining rung on the same broken input and reported as "every rung of the
    integrator ladder failed". Both must propagate as themselves, and neither may leave a retry behind."""
    H, psi0, times = _tiny_ladder_problem()
    with pytest.raises(ValueError, match="incompatible dimensions"):
        evolve(H, psi0, times, e_ops={"x": qt.qeye(3)})
    # 'diag' takes no atol/rtol/nsteps and raises KeyError before it integrates anything
    with pytest.raises(KeyError, match="not supported"):
        evolve(H, psi0, times, options=SolverOptions(integrators=("diag", "dop853")))


@pytest.mark.slow
def test_the_ladder_escalates_when_a_rung_fails_and_records_the_retry() -> None:
    """Audit E11: no test ever triggered an escalation, so nothing checked that a failing rung is recorded and the next rung
    carries the integration.

    The plan's own trigger cannot be used, because it does not reproduce here: on the Section 11.1 pulse at nsteps = 1e7
    ``dop853`` succeeds at d_m = 121, 151 and 201 at atol 1e-10 AND 1e-8 (measured; ``conv.integrator_ladder_stiffness`` in
    the ledger carries the numbers, and the M2 fixer found the same on the one-mode fixture). What is reproducible is the
    step budget: on the Section 11.1 row at dimension 256 (two modes, d_m = 8, 20 us, atol 1e-10, rtol 1e-8) the fifth-order
    ``tsit5`` needs more than 2000 steps per output interval and succeeds at 4000, while the eighth-order ``dop853`` needs
    more than 500 and succeeds at 1000. A ladder ``("tsit5", "dop853")`` at nsteps = 2000 therefore aborts on its first rung
    with the integrator's own ``IntegratorException`` and finishes on its second, with a factor two of margin on each side;
    the step counts are set by the integrators' arithmetic, not by the hardware."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "scripts"))
    from bench_ms_timing_v5 import build  # type: ignore[import-not-found]

    nmodes, d_m = 2, 8
    H = build(nmodes, d_m, "csr")[0]
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
    # the escalated run and the un-escalated one are the same evolution to what Section 11.1 means by "identical results":
    # at atol 1e-10, rtol 1e-8 the plan's own integrators agree to 1.3e-7 (dimension 48) to 5.8e-7 (2048) in norm, and the
    # fifth-order tsit5 against the eighth-order dop853 measures 4.41e-7 here at dimension 256
    assert (roomy.final - ev.final).norm() < 1e-6, (roomy.final - ev.final).norm()


# ---- E18: resolve_level, and the run-level convergence check of Section 5.5 --------------------------------------------------


def test_resolve_level_is_what_run_uses_and_its_estimate_path_works() -> None:
    """PLAN.md Appendix E: ``level="auto"`` "resolves through ``resolve_level(device, circuit, options)``". It was dead code
    while ``run()`` inlined ``within_budget`` (audit B6/E18). With the run's actual space the guards are exact; without one
    ``resolve_level`` estimates them for the Section 9.6 fixture rule (two resolved modes at d_m = 12), which nothing tested."""
    from qutip_trap.run.levels import ESTIMATE_MODE_DIMENSION, ESTIMATE_RESOLVED_MODES, resolve_level

    dev = chain_device(2)
    native = compile_to_native(BELL, dev)
    opts = SolverOptions()
    # the estimate: 2 ions, an entangling gate, so two resolved modes at d_m = 12 -> dimension 4 x 144 = 576
    assert (ESTIMATE_RESOLVED_MODES, ESTIMATE_MODE_DIMENSION) == (2, 12)
    assert resolve_level(dev, native, opts) == "JOINT_EXACT"
    assert resolve_level(dev, native, SolverOptions(joint_dimension_max=64)) == "GATE_LOCAL"
    nnz_estimate = 2 * (2**2) * (ESTIMATE_MODE_DIMENSION**2) ** ESTIMATE_RESOLVED_MODES
    assert resolve_level(dev, native, SolverOptions(nnz_max=nnz_estimate - 1)) == "GATE_LOCAL"
    assert resolve_level(dev, native, SolverOptions(nnz_max=nnz_estimate)) == "JOINT_EXACT"
    # a circuit with no entangling gate estimates ONE resolved mode
    single = compile_to_native(Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1)), dev)
    assert resolve_level(dev, single, SolverOptions(joint_dimension_max=48)) == "JOINT_EXACT"
    assert resolve_level(dev, single, SolverOptions(joint_dimension_max=47)) == "GATE_LOCAL"
    # with a space the verdict is the exact guard, not the estimate
    big = HilbertSpace((2, 2), (ModeTruncation(0, 40, (0, 4), 0.1),), None, tuple(range(1, 6)))
    assert resolve_level(dev, native, SolverOptions(joint_dimension_max=64), space=big) == "GATE_LOCAL"


@pytest.mark.slow
def test_run_reports_the_section_5_5_tolerance_convergence_when_asked() -> None:
    """Section 5.5's second bullet through ``run()``: ``SolverOptions.convergence_check`` repeats the evolution with atol and
    rtol tightened by ten and puts the change in the register populations on ``Diagnostics.convergence`` (M2's
    ``ConvergenceReport``, the run-path plumbing M9's). Off by default, and ``None`` then means "not asked for"."""
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=WINDOWS)
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


# ---- 9.9: mcsolve with and without improved_sampling against the mesolve histogram -----------------------------------------


@pytest.mark.slow
def test_mcsolve_with_and_without_improved_sampling_converge_to_the_mesolve_histogram() -> None:
    """The last clause of Section 9.9: "shots drawn from ``mcsolve`` with and without ``improved_sampling`` converge to the
    ``mesolve`` histogram on a two-ion dissipative case". ``improved_sampling`` appeared nowhere in the tree before the M6 fix
    and this comparison existed nowhere (audit E7).

    Two ions, one resolved mode at d_m = 12 and a heating channel, driven by ONE carrier pulse: dimension 48, so ``mesolve``
    is the exact reference, and one trajectory segment, which is the condition under which the engine applies improved
    sampling at all (``IMPROVED_SAMPLING_MULTI_SEGMENT``: the no-jump/jump split decomposes the WHOLE evolution, so a
    multi-segment schedule keeps uniform weights and says so). Both trajectory ensembles must land on the density matrix's
    register populations inside the multinomial band of their own trajectory count, both mixtures must have unit weight, and
    the improved-sampling ensemble must carry one member more than ``ntraj`` - the deterministic no-jump trajectory of weight
    p_no-jump, which is the bias the plan warns about when shots are drawn uniformly from the stored trajectories.
    """
    import dataclasses as _dc

    from qutip_trap.api import white_spectrum
    from tests.m4_fixtures import X_COM_TWO_IONS

    dev = chain_device(2)
    noisy = _dc.replace(
        dev,
        noise=_dc.replace(dev.noise, S_E=white_spectrum(2e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0),
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
