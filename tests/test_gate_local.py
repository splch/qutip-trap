"""GATE_LOCAL against JOINT_EXACT (PLAN.md Sections 5.4, 6.8, 9.8, 9.17): the step partition of a schedule, the exact
agreement on a single-qubit gate, the Bell circuit within the reported bound with the motional bookkeeping, the register as
a density matrix or a Kraus-sampled ensemble, the per-step register and channels, the map-accuracy rule on the trajectory
path, the caches, and the three- and four-ion circuits of Section 9.8 row 1 as slow tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation, compile_to_native
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule, schedule
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec, SolverOptions
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.dynamics.tomography import apply_kraus_dm, kraus_operators
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics
from qutip_trap.run.gate_local import (
    REGISTER_STORE_DIM_MAX,
    AppliedChannel,
    clear_gate_local_cache,
    gate_steps,
)
from qutip_trap.run.job import last_record, register_fidelity
from qutip_trap.run.levels import within_budget
from qutip_trap.run.space import best_contributions, select_space
from tests.fixtures import BELL, WINDOWS, run, single_ion_raman_device, two_ion_surrogate

GPI2 = Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1))
FAST = SolverOptions(branch_weight_min=1e-3)


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = yb171_chain(2)
    sur = two_ion_surrogate(1000)
    kw = dict(
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(FAST),
    )
    return fx, sur, kw


def _populations(res) -> np.ndarray:  # type: ignore[no-untyped-def]
    assert res.final_state is not None
    return np.real(np.diag(np.asarray(res.final_state.full())))


def test_gate_steps_partition_the_schedule_into_gates_and_idles(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Every pulse lies in exactly one gate step, the steps alternate with the dead-time idles, every GateTarget's pulses lie in one
    step, and the entangling gate as played is attached to its step."""
    fx, sur, _kw = two_ion
    sched = schedule(compile_to_native(BELL), fx.device, sur.table)
    steps = gate_steps(sched)
    gates = [s for s in steps if s.kind == "gate"]
    idles = [s for s in steps if s.kind == "idle"]
    assert len(gates) == 6 and len(idles) == 6, [s.gate_id for s in steps]
    assert all(steps[k].kind != steps[k + 1].kind for k in range(len(steps) - 1))
    covered = [p.gate_id for s in gates for p in s.pulses]
    assert sorted(covered) == sorted(p.gate_id for p in sched.pulses) and len(set(covered)) == len(covered)
    assert len(sched.targets) == 6 and all(len(s.targets) == 1 for s in gates)
    ms_steps = [s for s in gates if s.played]
    assert len(ms_steps) == 1 and ms_steps[0].played[0].gate_id == "ms[2]" and ms_steps[0].ions == (0, 1)
    assert ms_steps[0].gate_id == "ms[2]" and ms_steps[0].targets[0].native[0] == "ms"
    for k in range(len(steps) - 1):
        assert steps[k + 1].t_start_s == pytest.approx(steps[k].t_end_s, abs=1e-12)
    assert idles[0].duration_s == pytest.approx(fx.device.hardware.dead_time_s)
    assert steps[-1].t_end_s == pytest.approx(sched.pulses_end_s)


def test_single_qubit_gate_matches_joint_exact_to_solver_tolerance(two_ion) -> None:  # type: ignore[no-untyped-def]
    """A GPi2 on ion 0 with its addressing crosstalk onto ion 1: the gate-local space is the two ions with every mode frozen, the
    tomography (the propagator route: one segment propagator per Fock branch is the branch's Kraus operator, the sixteen inputs
    follow by linearity) returns a CPTP map whose application reproduces the JOINT_EXACT register to the solver tolerance; the
    channel summary sits at the crosstalk scale; a second run hits the cache, for the gate step and for the idle channels."""
    fx, sur, kw = two_ion
    clear_gate_local_cache()
    # the register of GATE_LOCAL is the pumped density matrix itself; JOINT_EXACT enumerates it into branches, so the comparison
    # keeps every branch (the preparation error is 2.5e-6, below the default cutoff) on this cheap all-frozen space, and the
    # tomography's tail rule is switched off for the same reason (the keyed tolerance and the derived margin never touch a
    # carrier step: no resolved mode)
    exact = {
        **kw,
        "numerics": Numerics.from_solver_options(
            SolverOptions(branch_weight_min=1e-9, tomography_dropped_weight_max=0.0)
        ),
    }
    a = run(GPI2, fx.device, 100, level="JOINT_EXACT", **exact)  # type: ignore[arg-type]
    b = run(GPI2, fx.device, 100, level="GATE_LOCAL", **exact)  # type: ignore[arg-type]
    assert a.diagnostics.level == "JOINT_EXACT" and b.diagnostics.level == "GATE_LOCAL"
    assert np.max(np.abs(np.asarray(a.final_state.full()) - np.asarray(b.final_state.full()))) < 1e-7
    gl = b.diagnostics.gate_local
    assert gl is not None and gl.register == "density_matrix" and gl.ensemble_size == 1
    gate = [s for s in gl.steps if s.kind == "gate"]
    assert len(gate) == 1
    s = gate[0]
    assert (
        s.ions == (0, 1)
        and s.space_dims == (2, 2)
        and s.resolved == ()
        and s.n_inputs == 16
        and s.n_branches >= 1
    )
    assert s.method == "sesolve" and s.n_traj == 1 and not s.cache_hit and s.engine_runs == s.n_branches
    assert s.route == "propagator" and all(st.route == "propagator" for st in gl.steps)
    assert gl.idle_cache_hits == 0, "the first idle of a fresh cache computes both ions' channels"
    # the floor at 1e-9 drops a few 1e-10 of weight (reported as 2w); no tail rule here, no keyed tolerance on a carrier step
    assert 0.0 <= s.branch_error_bound < 1e-8 and s.tolerance_change == 0.0 and s.tolerances == (1e-10, 1e-8)
    assert gl.branch_error_total < 1e-8 and gl.tolerance_change_total == 0.0 and s.element_error == {}
    assert s.tp_residual < 1e-10 and s.cp_residual < 1e-10, "Section 9.17: both residuals reported and tiny"
    assert s.summary is not None
    eps = sur.table.crosstalk[(0, 1)].value
    crosstalk_flip = math.sin(eps * math.pi / 4.0) ** 2
    assert 0.1 * crosstalk_flip < s.summary.average_gate_infidelity < 3.0 * crosstalk_flip + 1e-4
    assert abs(sum(s.summary.pauli_twirled.values()) - 1.0) < 1e-9 and s.summary.pauli_twirled["II"] > 0.99
    assert s.residual_bound == 0.0 and s.frozen_coupled == (2, 3), (
        "the x modes are frozen for a carrier pulse"
    )
    # the reported bound is the frozen modes' off-resonant excitation under the carrier, (eta Omega/omega_m)^2 summed over the
    # nearest sidebands (Section 5.2), a few 1e-5: JOINT_EXACT freezes the same modes, so the two registers agree far inside it
    assert gl.residual_bound_total == 0.0 and gl.dropped_crosstalk_total < 1e-6
    assert 1e-6 < gl.frozen_excitation_total < 1e-4 and gl.discrepancy_bound == pytest.approx(
        gl.frozen_excitation_total + gl.branch_error_total
    )
    assert any("GATE_LOCAL" in x for x in b.diagnostics.approximations)
    assert (
        b.probabilities.keys() <= {"00", "01", "10", "11"}
        and abs(register_fidelity(a) - register_fidelity(b)) < 1e-7
    )
    c = run(GPI2, fx.device, 100, level="GATE_LOCAL", **exact)  # type: ignore[arg-type]
    assert c.diagnostics.gate_local is not None and c.diagnostics.gate_local.cache_hits >= 1
    assert c.diagnostics.gate_local.idle_cache_hits == 2 and c.diagnostics.gate_local.engine_runs == 0
    assert [st.cache_hit for st in c.diagnostics.gate_local.steps] == [True, True]
    assert np.array_equal(b.bitstrings, c.bitstrings)
    rec = last_record(b)
    assert rec.gate_local is gl and rec.traces == () and rec.register_state is not None


@pytest.mark.slow
def test_bell_circuit_gate_local_matches_joint_exact_within_the_reported_bound(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 9.8: the Bell circuit through GATE_LOCAL (the MS step on the exact two-ion, two-mode space from the tracked thermal
    state) against JOINT_EXACT: the register populations agree within the residual-displacement bound the run reports, the tracked
    occupations match the joint run's reduced motional state to 5%, the channel summary of the MS step sits inside the
    intrinsic budget, and the Choi matrix is trace preserving to 1e-10."""
    fx, sur, kw = two_ion
    a = run(BELL, fx.device, 300, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    b = run(BELL, fx.device, 300, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    pa, pb = _populations(a), _populations(b)
    gl = b.diagnostics.gate_local
    assert gl is not None
    # the bound plus the weight of the initial-mixture branches JOINT_EXACT dropped (GATE_LOCAL keeps the full register),
    # no slack on top: Section 9.8 row 1 says "disagreement above it fails"
    bound = gl.discrepancy_bound + a.diagnostics.dropped_branch_weight
    assert gl.discrepancy_bound > 0.0
    assert np.max(np.abs(pa - pb)) < bound, (pa, pb, bound)
    assert abs(register_fidelity(a) - register_fidelity(b)) < bound
    ms = [s for s in gl.steps if s.kind == "gate" and s.resolved]
    assert len(ms) == 1
    s = ms[0]
    assert s.resolved == (2, 3) and s.space_dims[:2] == (2, 2) and s.n_inputs == 16 and s.n_branches >= 1
    assert s.tp_residual < 1e-10 and s.cp_residual < 1e-8
    # the isometry route, the keyed tolerance and the branch floor, each with its reported term in the bound
    assert s.route == "isometry" and s.tolerances == (1e-8, 1e-6) and s.tolerance_change > 0.0
    assert s.branch_error_bound >= 0.0 and s.branch_error_bound <= 2.0 * 1e-3 / 4 + 2.0 * 1e-3
    assert set(s.element_error) == {2, 3} and all(0.0 <= v <= 1e-8 for v in s.element_error.values())
    assert (
        gl.tolerance_change_total == pytest.approx(s.tolerance_change)
        and gl.branch_error_total >= s.branch_error_bound
    )
    assert gl.discrepancy_bound == pytest.approx(
        gl.residual_bound_total
        + gl.frozen_excitation_total
        + gl.dropped_crosstalk_total
        + gl.branch_error_total
        + gl.tolerance_change_total
    )
    # the derived margin: caps two to three levels below the fixture's, at the declared element tolerance 1e-8
    assert all(d <= 11 for d in s.space_dims[2:]), s.space_dims
    assert any("derived for interior elements exact to 1e-08" in n for n in s.notes)
    assert (
        s.summary is not None
        and 0.0 < s.summary.average_gate_infidelity < b.diagnostics.intrinsic_budget["total"]
    )
    assert s.summary.depolarizing_rate == pytest.approx(1.25 * s.summary.average_gate_infidelity, rel=1e-9)
    assert set(s.residual_displacement) == {2, 3} and s.residual_bound > 0.0
    # the reduced motional state after the gate is a mixture: the thermal input alone has 1 - Tr rho^2 = 2 nbar/(1 + 2 nbar)
    # (0.04 at nbar = 0.02), and the spin-conditioned displacements add little to it
    for m, v in s.purity_deficit.items():
        nb = s.nbar_after[m]
        assert 0.5 * (2.0 * nb / (1.0 + 2.0 * nb)) <= v < 0.1, (m, v, nb)
    # Section 9.8 row 2: the tracked nbar after the gate against the joint run's reduced motional state at the gate's end
    # the branch-weighted average over the initial mixture's branches (one trace per branch)
    rec_a = last_record(a)
    w_tot = sum(br.weight for br in rec_a.branches)
    for m in (2, 3):
        joint_n = (
            sum(br.weight * tr.final.motional.nbar[m] for br, tr in zip(rec_a.branches, rec_a.traces)) / w_tot
        )
        tracked = s.nbar_after[m]
        assert abs(tracked - joint_n) <= 0.05 * max(joint_n, 1e-3), (m, tracked, joint_n)
    # the sampled histograms agree within their error bars
    for key in ("00", "11"):
        assert (
            abs(a.probabilities.get(key, 0.0) - b.probabilities.get(key, 0.0))
            < 4.0 * a.error_bars[key] + bound
        )


def test_ensemble_register_by_kraus_sampling_agrees_with_the_density_matrix(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Above ``register_dm_max_qubits`` the register is a pure-state ensemble and every map is applied by Kraus sampling
    (Section 5.4); forced on the two-ion GPi2 circuit it reproduces the density-matrix histogram within statistics."""
    fx, sur, kw = two_ion
    opts = SolverOptions(branch_weight_min=1e-3, register_dm_max_qubits=1, register_ensemble=24)
    b = run(
        GPI2, fx.device, 240, level="GATE_LOCAL", **{**kw, "numerics": Numerics.from_solver_options(opts)}
    )  # type: ignore[arg-type]
    a = run(GPI2, fx.device, 240, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    gl = b.diagnostics.gate_local
    assert gl is not None and gl.register == "ensemble" and gl.ensemble_size == 24
    assert b.final_state is None and any("Kraus sampling" in n for n in b.diagnostics.approximations)
    for key in ("00", "10"):
        assert abs(a.probabilities.get(key, 0.0) - b.probabilities.get(key, 0.0)) < 0.12
    assert b.shots == 240


def test_map_accuracy_rule_fixes_the_trajectory_count_on_the_trajectory_path() -> None:
    """Section 5.4: on the trajectory path every tomography input is propagated with n_traj = ceil(1/epsilon_map); with mesolve
    (the deterministic cross-check below the dimension threshold) the count is one. A one-ion, one-mode space with heating."""
    import dataclasses

    dev = single_ion_raman_device()
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-11, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    from qutip_trap.light.raman import derive_raman_drive, square_drive

    dd = derive_raman_drive(noisy, 0, (0, 1), scattering=False)
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, 0.5 * math.pi / om, "p", ())
    space = HilbertSpace((2,), (ModeTruncation(1, 6, (0, 1), 0.2),), None, (0, 2))
    model = MotionalModel(reduced={}, nbar={0: 0.0, 1: 0.1, 2: 0.0}, frozen=(0, 2))
    eng = JointExactEngine(device_channels=True)
    rec_me = eng.tomography(
        noisy, pulse, space, model, quiet_sample(), SeedSpec(0), SolverOptions(lindblad_method="mesolve")
    )
    assert rec_me.method == "mesolve" and rec_me.n_traj == 1 and rec_me.tp_residual < 1e-10
    assert len(rec_me.labels) == 4 and rec_me.branches >= 1
    opts = SolverOptions(lindblad_method="mcsolve", map_accuracy=0.25, branch_weight_min=0.05)
    rec_mc = eng.tomography(noisy, pulse, space, model, quiet_sample(), SeedSpec(0), opts)
    # Section 9.17's rule is ceil(1/eps_map) = 4 STOCHASTIC trajectories per input; under improved_sampling (Section 5.3's
    # default) the no-jump trajectory is one more, deterministic member of weight p_no-jump, so the mixture has five
    plain = dataclasses.replace(opts, improved_sampling=False)
    rec_plain = eng.tomography(noisy, pulse, space, model, quiet_sample(), SeedSpec(0), plain)
    assert rec_mc.method == "mcsolve" and rec_plain.method == "mcsolve"
    assert rec_plain.n_traj == 4 and rec_mc.n_traj == 5
    assert rec_mc.tp_residual < 1e-10 and rec_mc.cp_residual < 1e-8, (
        "the projection restores CPTP on the noisy reconstruction"
    )
    # the trajectory estimate approaches the deterministic map: the Choi matrices agree to a few percent at four trajectories
    assert np.max(np.abs(rec_mc.choi - rec_me.choi)) < 0.15
    summary = rec_me.summary(None)
    assert (
        math.isnan(summary.average_gate_infidelity) and abs(sum(summary.pauli_twirled.values()) - 1.0) < 1e-9
    )


def test_every_step_reports_its_register_and_the_channels_compose_it(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Every step of the first sample's walk reports the register after it and the channels it applied; composing a step's
    channels on the previous step's register gives its own, and the last equals the run's recombined register."""
    fx, _sur, kw = two_ion
    rec = last_record(run(BELL, fx.device, 100, level="GATE_LOCAL", **kw))  # type: ignore[arg-type]
    assert rec.gate_local is not None
    steps = rec.gate_local.steps
    assert steps and 4 <= REGISTER_STORE_DIM_MAX
    for st in steps:
        assert st.register_after is not None and st.register_after.shape == (4, 4)
        assert abs(np.trace(st.register_after) - 1.0) < 1e-9
        assert np.allclose(st.register_after, st.register_after.conj().T, atol=1e-12)
        assert st.channels and all(isinstance(c, AppliedChannel) for c in st.channels)
        if st.kind == "gate":
            assert len(st.channels) == 1 and st.channels[0].ions == st.ions
            assert st.summary is not None and np.allclose(st.channels[0].choi, st.summary.choi, atol=1e-12)
        else:
            assert [c.ions for c in st.channels] == [(0,), (1,)]
            assert all(c.choi.shape == (4, 4) for c in st.channels)
    for before, after in zip(steps, steps[1:]):
        rho = np.asarray(before.register_after, dtype=complex)
        for ch in after.channels:
            rho = apply_kraus_dm(rho, kraus_operators(ch.choi), [2, 2], list(ch.ions))
        assert after.register_after is not None
        assert np.max(np.abs(rho - after.register_after)) < 1e-10, after.gate_id
    assert rec.register_state is not None
    assert np.max(np.abs(steps[-1].register_after - np.asarray(rec.register_state.full()))) < 1e-10


def test_idle_channel_of_a_detuned_qubit_is_the_phase_rotation(two_ion) -> None:  # type: ignore[no-untyped-def]
    """An idle Schedule through the tomography: a quasi-static qubit offset makes the one-qubit channel the unitary
    e^{-i pi delta t sigma_z} (the closed-form segment), the map is unitary (one Kraus operator) and trace preserving."""
    from qutip_trap.noise.sampling import NoiseSample, key_qubit_offset_hz

    fx, _sur, _kw = two_ion
    dev = fx.device
    n_modes = len(dev.crystal.modes)
    space = HilbertSpace((2,), (), None, tuple(range(n_modes)), ions=(1,))
    t0, tau = 1e-4, 2.5e-4
    sched = Schedule((), ((t0, t0 + tau),), (), {}, t0_s=t0)
    sample = NoiseSample(0, {key_qubit_offset_hz(1): 400.0}, {})
    model = MotionalModel(reduced={}, nbar={m: 0.0 for m in range(n_modes)}, frozen=tuple(range(n_modes)))
    rec = JointExactEngine().tomography(dev, sched, space, model, sample, SeedSpec(0), SolverOptions())
    ks = rec.kraus()
    assert len(ks) == 1 and rec.tp_residual < 1e-10
    assert rec.route == "propagator" and rec.engine_runs == 1 and rec.integrators == ("exact",)
    theta = 2.0 * math.pi * 400.0 * tau
    ideal = np.diag(
        [np.exp(0.5j * theta), np.exp(-0.5j * theta)]
    )  # (delta/2) sigma_z with sigma_z = |1><1| - |0><0|
    phase = ks[0][0, 0] / ideal[0, 0]
    assert abs(abs(phase) - 1.0) < 1e-9 and np.max(np.abs(ks[0] - phase * ideal)) < 1e-7
    assert rec.summary(ideal).average_gate_infidelity < 1e-10


def _compare_levels(a, b):  # type: ignore[no-untyped-def]
    """Section 9.8 rows 1 and 2 on a JOINT_EXACT run ``a`` and a GATE_LOCAL run ``b`` of one circuit: the final register
    populations agree within the bound GATE_LOCAL reports plus the initial-mixture weight JOINT_EXACT dropped (no other
    slack), and the tracked occupation after every entangling step is the joint run's branch-weighted reduced state at that
    time to 5 %. Returns (the bound, the entangling steps)."""
    gl = b.diagnostics.gate_local
    assert gl is not None and gl.discrepancy_bound > 0.0
    bound = gl.discrepancy_bound + a.diagnostics.dropped_branch_weight
    pa, pb = _populations(a), _populations(b)
    assert np.max(np.abs(pa - pb)) < bound, (np.max(np.abs(pa - pb)), gl.discrepancy_bound, bound)
    assert abs(register_fidelity(a) - register_fidelity(b)) < bound
    ms_steps = [s for s in gl.steps if s.kind == "gate" and s.resolved]
    rec_a = last_record(a)
    w_tot = sum(br.weight for br in rec_a.branches)
    for s in ms_steps:
        k = int(np.argmin(np.abs(rec_a.traces[0].times_s - s.t_end_s)))
        for m in s.resolved:
            joint_n = float(
                sum(
                    br.weight * np.real(tr.mode_occupations[m][k])
                    for br, tr in zip(rec_a.branches, rec_a.traces)
                )
                / w_tot
            )
            assert abs(s.nbar_after[m] - joint_n) <= 0.05 * max(joint_n, 1e-3) + 1e-4, (
                s.gate_id,
                m,
                s.nbar_after[m],
                joint_n,
            )
    return bound, ms_steps


@pytest.mark.slow
def test_three_ion_ghz_circuit_gate_local_against_joint_exact() -> None:
    """Section 9.8 row 1: the three-ion GHZ circuit (two entangling gates, five carrier pulses), with the crosstalk
    neighbours inside the gate-local spaces."""
    fx = yb171_chain(3, address_waist_m=2.0e-6)
    sur = surrogate_table(
        fx.device, pairs=[(0, 1), (1, 2)], detection_records=1000, detection_windows_s=WINDOWS
    )
    ghz = Circuit(
        3, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()), Operation("cnot", (1, 2), ())), (0, 1, 2)
    )
    kw = dict(
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    a = run(ghz, fx.device, 500, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    b = run(ghz, fx.device, 500, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    _bound, ms_steps = _compare_levels(a, b)
    assert len(ms_steps) == 2 and all(s.ions in ((0, 1), (1, 2), (0, 1, 2)) for s in ms_steps)
    assert b.probabilities.get("000", 0.0) + b.probabilities.get("111", 0.0) > 0.9
    assert register_fidelity(b) > 0.9


GHZ4 = Circuit(
    4,
    (
        Operation("h", (0,), ()),
        Operation("cnot", (0, 1), ()),
        Operation("cnot", (1, 2), ()),
        Operation("cnot", (2, 3), ()),
    ),
    (0, 1, 2, 3),
)


@pytest.mark.slow
def test_four_ion_ghz_circuit_gate_local_against_joint_exact() -> None:
    """Section 9.8 row 1 on four ions: GHZ4 has three Mølmer-Sørensen steps, and at ``freeze_chi_max_rad = 0.3`` the gate
    mode (chi 0.81 to 0.84 rad per step, the other x modes at most 0.13) is the only resolved mode at BOTH levels, so the
    comparison measures GATE_LOCAL's own approximation: JOINT_EXACT runs at dimension 192, and the three frozen x modes, whose
    chi the common surrogate table absorbs, carry their Debye-Waller factors and Fock branches at both levels
    (``conv.four_ion_gate_local_fixture``)."""
    fx = yb171_chain(4)
    sur = surrogate_table(
        fx.device, pairs=[(0, 1), (1, 2), (2, 3)], detection_records=200, detection_windows_s=WINDOWS
    )
    sched = schedule(compile_to_native(GHZ4), fx.device, sur.table, t0_s=0.0)
    nbar0 = {m: 0.0 for m in range(len(fx.device.crystal.modes))}
    best = best_contributions(fx.device, sched.gates, nbar0)
    opts = SolverOptions(freeze_chi_max_rad=0.3)
    selection = select_space(fx.device, sched, opts, nbar=nbar0, caps={7: 12})
    resolved = selection.resolved_modes
    assert resolved == (7,), (resolved, {m: round(c.chi_rad, 4) for m, c in sorted(best.items())})
    assert selection.space.dims == [2, 2, 2, 2, 12] and selection.space.dimension == 192
    inside, dim, _nnz = within_budget(selection.space, opts)
    assert inside and dim == 192, "JOINT_EXACT must be affordable inside the Section 11.5 guard"
    assert [selection.mode_class[m] for m in (4, 5, 6)] == ["frozen"] * 3, selection.mode_class
    kw = dict(
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(opts, caps={7: 12}),
        seed=3,
    )
    a = run(GHZ4, fx.device, 200, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    b = run(GHZ4, fx.device, 200, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    assert a.diagnostics.space.dimension == 192
    assert tuple(a.diagnostics.mode_class[m] for m in resolved) == ("resolved",) * len(resolved)
    _bound, ms_steps = _compare_levels(a, b)
    assert len(ms_steps) == 3, [s.gate_id for s in ms_steps]


# ---- the register bookkeeping and the idle cache ------------------------------------------------------------------------------


def test_register_marginal_is_the_partial_trace_in_the_requested_factor_order() -> None:
    """``Register.marginal`` reads the strided view of the density matrix; against QuTiP's ``ptrace`` (which returns the kept
    factors in ascending order) with the factors requested in a non-ascending order, and the ensemble path against the average."""
    from qutip_trap.run.gate_local import Register

    rng = np.random.default_rng(11)
    dims = (2, 3, 2)
    total = int(np.prod(dims))
    kets = []
    for _ in range(3):
        v = rng.normal(size=total) + 1j * rng.normal(size=total)
        kets.append(v / np.linalg.norm(v))
    rho = sum(np.outer(k, k.conj()) for k in kets) / len(kets)
    reg = Register.from_density_matrix(rho, dims)
    ens = Register(dims, kets=list(kets))
    swap = np.zeros((4, 4))
    swap[0, 0] = swap[3, 3] = swap[1, 2] = swap[2, 1] = 1.0
    full = qt.Qobj(rho, dims=[list(dims), list(dims)])
    for factors in ((0,), (1,), (2, 0), (0, 2), (1, 2)):
        m = reg.marginal(factors)
        m_ens = ens.marginal(factors)
        ref = np.asarray(full.ptrace(sorted(factors)).full())
        if list(factors) != sorted(factors):
            ref = swap @ ref @ swap.T
        assert m.shape == ref.shape and np.max(np.abs(m - ref)) < 1e-14, factors
        assert np.max(np.abs(m_ens - ref)) < 1e-14, factors
        assert abs(np.trace(m) - 1.0) < 1e-13


def test_idle_channels_are_cached_across_equal_dead_times_and_one_engine_serves_the_walk(
    two_ion, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Three carrier gates leave three dead-time idles of the same length: the six one-ion idle channels cost two extractions
    and four cache hits, the walk builds ONE engine, and with the cache defeated (a unique key per call) the register is the same
    to round-off and the run costs four more engine calls. The key is blind to the absolute window unless the sample carries a fast
    trajectory (an OU grid indexed by absolute time), and it sees the duration, the ion and the options."""
    import uuid

    from qutip_trap.noise.sampling import NoiseSample, key_qubit_trajectory_hz
    from qutip_trap.run import gate_local
    from qutip_trap.run.gate_local import EngineSetup, GateStep, _idle_key

    fx, sur, kw = two_ion
    circ = Circuit(
        2,
        (Operation("gpi2", (0,), (0.0,)), Operation("gpi2", (1,), (0.5,)), Operation("gpi2", (0,), (1.0,))),
        (0, 1),
    )
    built = 0
    original = EngineSetup.engine

    def counting(self):  # type: ignore[no-untyped-def]
        nonlocal built
        built += 1
        return original(self)

    monkeypatch.setattr(EngineSetup, "engine", counting)
    clear_gate_local_cache()
    a = run(circ, fx.device, 120, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    gl = a.diagnostics.gate_local
    assert gl is not None and built == 1, "one JOINT_EXACT engine per walk"
    idles = [s for s in gl.steps if s.kind == "idle"]
    assert len(idles) == 3 and gl.idle_cache_hits == 4 and gl.cache_hits == 0
    assert [s.cache_hit for s in idles] == [False, True, True]
    assert all(s.route == "propagator" for s in gl.steps)
    assert all(
        abs((i.t_end_s - i.t_start_s) - (idles[0].t_end_s - idles[0].t_start_s)) < 1e-15 for i in idles
    )
    runs_cached = gl.engine_runs
    monkeypatch.setattr(gate_local, "_idle_key", lambda *args, **kwargs: uuid.uuid4().hex)
    clear_gate_local_cache()
    b = run(circ, fx.device, 120, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    gl_b = b.diagnostics.gate_local
    assert gl_b is not None and gl_b.idle_cache_hits == 0 and gl_b.engine_runs == runs_cached + 4
    assert np.max(np.abs(np.asarray(a.final_state.full()) - np.asarray(b.final_state.full()))) < 1e-14
    assert np.array_equal(a.bitstrings, b.bitstrings)
    monkeypatch.undo()
    # the key itself
    dev = fx.device
    setup = EngineSetup(table=sur.table)
    opts = SolverOptions()
    quiet = quiet_sample()
    step_a = GateStep("idle", 1.0e-6, 2.0e-6, (), (), (), ())
    step_b = GateStep("idle", 7.3e-5, 7.4e-5, (), (), (), ())
    step_c = GateStep("idle", 1.0e-6, 2.5e-6, (), (), (), ())
    k = lambda step, smp, o=opts, q=0: _idle_key(dev, q, 2, step, smp, SeedSpec(0), o, setup)  # noqa: E731
    assert k(step_a, quiet) == k(step_b, quiet), "the same dead time anywhere in the walk is the same channel"
    assert k(step_a, quiet) != k(step_c, quiet) and k(step_a, quiet) != k(step_a, quiet, q=1)
    assert k(step_a, quiet) != k(step_a, quiet, o=SolverOptions(atol=1e-9))
    grid = np.array([np.linspace(0.0, 2e-4, 5), [10.0, -5.0, 3.0, 0.0, 1.0]])
    noisy = NoiseSample(0, {}, {key_qubit_trajectory_hz(0): grid})
    assert k(step_a, noisy) != k(step_b, noisy), (
        "a fast trajectory makes the channel depend on where the idle sits"
    )
    assert k(step_a, noisy) != k(step_a, quiet)
