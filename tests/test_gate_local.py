"""GATE_LOCAL against JOINT_EXACT (PLAN.md Section 5.4; Section 9.8; Section 9.17 rows "Size guard" and "GATE_LOCAL tomography";
Section 6.8; M9a): the step partition of a schedule, the exact agreement on a single-qubit gate, the Bell circuit within the
reported residual-displacement bound with the motional bookkeeping, the register as a density matrix or a Kraus-sampled ensemble,
the map-accuracy rule on the trajectory path, the cache, and the three-ion circuits of Section 9.8 as a slow test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import (
    Circuit,
    Operation,
    SolverOptions,
    gate_steps,
    last_record,
    register_fidelity,
    run,
    schedule,
    white_spectrum,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.run.gate_local import clear_gate_local_cache
from tests.m2_fixtures import single_ion_raman_device
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
GPI2 = Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = SolverOptions(branch_weight_min=1e-3)


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(
        fx.device,
        pairs=[(0, 1)],
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        detection_records=1000,
        detection_windows_s=WINDOWS,
    )
    kw = dict(
        table=sur.table,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        keep_final_state=True,
        options=FAST,
    )
    return fx, sur, kw


def _populations(res) -> np.ndarray:  # type: ignore[no-untyped-def]
    assert res.final_state is not None
    return np.real(np.diag(np.asarray(res.final_state.full())))


def test_gate_steps_partition_the_schedule_into_gates_and_idles(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Every pulse lies in exactly one gate step, the steps alternate with the dead-time idles, every GateTarget's pulses lie in one
    step, and the entangling gate as played is attached to its step."""
    fx, sur, _kw = two_ion
    from qutip_trap.control.compiler import compile_to_native

    sched = schedule(
        compile_to_native(BELL, fx.device),
        fx.device,
        sur.table,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
    )
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
    sixteen-input tomography returns a CPTP map whose application reproduces the JOINT_EXACT register to the solver tolerance; the
    channel summary sits at the crosstalk scale; a second run hits the cache."""
    fx, sur, kw = two_ion
    clear_gate_local_cache()
    # the register of GATE_LOCAL is the pumped density matrix itself; JOINT_EXACT enumerates it into branches, so the comparison
    # keeps every branch (the preparation error is 2.5e-6, below the default cutoff) on this cheap all-frozen space
    exact = {**kw, "options": SolverOptions(branch_weight_min=1e-9)}
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
    assert s.method == "sesolve" and s.n_traj == 1 and not s.cache_hit and s.engine_runs == 16 * s.n_branches
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
        gl.frozen_excitation_total
    )
    assert any("GATE_LOCAL" in x for x in b.diagnostics.approximations)
    assert (
        b.probabilities.keys() <= {"00", "01", "10", "11"}
        and abs(register_fidelity(a) - register_fidelity(b)) < 1e-7
    )
    c = run(GPI2, fx.device, 100, level="GATE_LOCAL", **exact)  # type: ignore[arg-type]
    assert c.diagnostics.gate_local is not None and c.diagnostics.gate_local.cache_hits >= 1
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
    # the bound plus the weight of the initial-mixture branches JOINT_EXACT dropped (GATE_LOCAL keeps the full register)
    bound = gl.discrepancy_bound + a.diagnostics.dropped_branch_weight
    assert gl.discrepancy_bound > 0.0
    assert np.max(np.abs(pa - pb)) < bound + 1e-6, (pa, pb, bound)
    assert abs(register_fidelity(a) - register_fidelity(b)) < bound + 1e-6
    ms = [s for s in gl.steps if s.kind == "gate" and s.resolved]
    assert len(ms) == 1
    s = ms[0]
    assert s.resolved == (2, 3) and s.space_dims[:2] == (2, 2) and s.n_inputs == 16 and s.n_branches >= 1
    assert s.tp_residual < 1e-10 and s.cp_residual < 1e-8
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
    b = run(GPI2, fx.device, 240, level="GATE_LOCAL", **{**kw, "options": opts})  # type: ignore[arg-type]
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
    assert rec_mc.method == "mcsolve" and rec_mc.n_traj == 4
    assert rec_mc.tp_residual < 1e-10 and rec_mc.cp_residual < 1e-8, (
        "the projection restores CPTP on the noisy reconstruction"
    )
    # the trajectory estimate approaches the deterministic map: the Choi matrices agree to a few percent at four trajectories
    assert np.max(np.abs(rec_mc.choi - rec_me.choi)) < 0.15
    summary = rec_me.summary(None)
    assert (
        math.isnan(summary.average_gate_infidelity) and abs(sum(summary.pauli_twirled.values()) - 1.0) < 1e-9
    )


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
    theta = 2.0 * math.pi * 400.0 * tau
    ideal = np.diag(
        [np.exp(0.5j * theta), np.exp(-0.5j * theta)]
    )  # (delta/2) sigma_z with sigma_z = |1><1| - |0><0|
    phase = ks[0][0, 0] / ideal[0, 0]
    assert abs(abs(phase) - 1.0) < 1e-9 and np.max(np.abs(ks[0] - phase * ideal)) < 1e-7
    assert rec.summary(ideal).average_gate_infidelity < 1e-10


@pytest.mark.slow
def test_three_ion_ghz_circuit_gate_local_against_joint_exact() -> None:
    """Section 9.8 row 1: the three-ion GHZ circuit (two entangling gates, five carrier pulses) through GATE_LOCAL against
    JOINT_EXACT: final probabilities within the reported bound, the tracked occupations within 5 % of the joint reduced state
    after every gate, the crosstalk neighbours inside the gate-local spaces."""
    fx = circuit_fixture(3, address_waist_m=2.0e-6)
    sur = surrogate_table(
        fx.device,
        pairs=[(0, 1), (1, 2)],
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        detection_records=1000,
        detection_windows_s=WINDOWS,
    )
    ghz = Circuit(
        3, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()), Operation("cnot", (1, 2), ())), (0, 1, 2)
    )
    kw = dict(
        table=sur.table,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        keep_final_state=True,
        options=SolverOptions(branch_weight_min=1e-2),
    )
    a = run(ghz, fx.device, 500, level="JOINT_EXACT", **kw)  # type: ignore[arg-type]
    b = run(ghz, fx.device, 500, level="GATE_LOCAL", **kw)  # type: ignore[arg-type]
    pa, pb = _populations(a), _populations(b)
    gl = b.diagnostics.gate_local
    assert gl is not None
    assert np.max(np.abs(pa - pb)) < gl.discrepancy_bound + 1e-5, (
        np.max(np.abs(pa - pb)),
        gl.discrepancy_bound,
    )
    ms_steps = [s for s in gl.steps if s.kind == "gate" and s.resolved]
    assert len(ms_steps) == 2 and all(s.ions in ((0, 1), (1, 2), (0, 1, 2)) for s in ms_steps)
    rec_a = last_record(a)
    w_tot = sum(br.weight for br in rec_a.branches)
    # the joint occupations at each gate's end, branch-weighted over the initial mixture: the traces store the segment endpoints
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
    assert b.probabilities.get("000", 0.0) + b.probabilities.get("111", 0.0) > 0.9
    assert (
        register_fidelity(b) > 0.9
        and abs(register_fidelity(a) - register_fidelity(b)) < gl.discrepancy_bound + 1e-3
    )
