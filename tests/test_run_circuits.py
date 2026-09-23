"""End-to-end circuits in JOINT_EXACT (PLAN.md Section 9.6; Sections 3.4, 5.2, 5.7, 7.2, 7.3, 8.6): the Bell state with all physics
on against the ideal distribution and the intrinsic budget, the diagnostics, seeds, the two readout paths, the surrogate
calibration path, and the three- and four-ion rows (GHZ, Wright's crosstalk model) as slow tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.calibration import calibrate
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation, ideal_probabilities
from qutip_trap.control.schedule import ScheduleError
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.job import enumerate_branches, last_record, register_fidelity, run
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = SolverOptions(branch_weight_min=1e-4)


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=2000, detection_windows_s=WINDOWS)
    return fx, sur


@pytest.fixture(scope="module")
def bell(two_ion):  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    res = run(
        BELL,
        fx.device,
        2000,
        table=sur.table,
        keep_final_state=True,
        physics=Physics.from_solver_options(FAST),
        numerics=Numerics.from_solver_options(FAST),
    )
    return fx, sur, res


def test_surrogate_table_carries_seeds_spot_checked_waveform_and_detection(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5: derived values as seed entries, the pair's waveform exactly calibrated on the resolved-mode space, the
    detection threshold and window from the simulated readout model; the scheduler reads only this table."""
    fx, sur = two_ion
    t = sur.table
    assert t.surrogate and t.device_hash == fx.device.hash()
    assert all(e.status == "seed" for e in t.rabi.values()) and set(t.rabi) == {
        (0, 2),
        (1, 4),
        (0, 0),
        (1, 0),
    }, (
        "the addressing pairs' and the global pair's carrier Rabi frequencies (the MS gate's light shift is compensated from it)"
    )
    # 100.9 kHz until the M0a fix of 2026-09-08 (audit item E4) gave the 171Yb+ Raman intermediate sum its
    # P3/2 path: for this fixture's polarization the two paths ADD (the P3/2 partial sum is +0.4577 of the
    # P1/2 one), so every carrier Rabi seed rose by the same factor 1.4577 -> 146.88 and 147.12 kHz.
    assert all(abs(e.value - 147.0e3) < 2e3 for e in t.rabi.values())
    assert t.stark[(0, 0)].status == "seed" and abs(t.stark[(0, 0)].value) < 100.0
    assert all(e.status == "seed" for e in t.qubit_freq.values()) and len(t.qubit_freq) == 2
    assert 0.01 < t.crosstalk[(0, 1)].value < 0.04, "Wright's 1-4% addressing crosstalk from the 2.5 um waist"
    assert t.crosstalk[(0, 1)].value == pytest.approx(t.crosstalk[(1, 0)].value)
    wf = t.waveform_for((0, 1))
    assert wf is not None and wf.phi_s.status == "calibrated" and wf.phi_s.experiment == "exact_spot_check"
    assert abs(abs(wf.chi_total_rad) - math.pi / 4.0) < 2e-4, (
        "the calibrated waveform carries its exact |angle| (signed by the detuning side)"
    )
    run0 = sur.entangling[(0, 1)]
    assert run0.converged and run0.checks[-1].fidelity > 0.999 and run0.checks[-1].leakage < 3e-4
    assert sur.mode_classes[(0, 1)] == {2: "resolved", 3: "resolved"}
    for key in ("threshold", "window_s", "eps_B", "eps_D"):
        assert t.detection[key].status == "calibrated"
    assert 10e-6 <= t.detection["window_s"].value <= 40e-6
    assert 1e-4 < t.detection["eps_B"].value < 5e-3 and 1e-4 < t.detection["eps_D"].value < 5e-3
    assert set(t.nbar) == set(range(6)) and t.nbar[3].value < 0.05 and t.nbar[0].value > 1.0


def test_bell_state_probabilities_match_the_ideal_distribution_within_readout_and_statistics(bell) -> None:  # type: ignore[no-untyped-def]
    """Section 9.6: the two-ion Bell state with all physics on; populations as in Section 7.9 (P_00 ~ P_11, the odd strings at
    the readout-error level), the ideal statevector distribution beside them."""
    fx, sur, res = bell
    p = res.probabilities
    ideal = ideal_probabilities(BELL)
    assert ideal == pytest.approx({"00": 0.5, "11": 0.5})
    assert res.shots == 2000 and res.n_qubits == 2 and res.bit_order == "qubit0_lsb"
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["00"] + p["11"] > 0.99
    assert abs(p["00"] - p["11"]) < 5.0 * res.error_bars["00"]
    assert p.get("01", 0.0) + p.get("10", 0.0) < 0.01
    assert res.error_bars["00"] == pytest.approx(math.sqrt(p["00"] * (1.0 - p["00"]) / 2000.0))
    assert (
        set(res.to_ionq_v1_probabilities()) <= {"0", "1", "2", "3"}
        and res.to_ionq_v1_probabilities()["0"] == p["00"]
    )
    assert res.heralds.shape == (2000,) and res.discarded_shots == 0 and res.photon_records is None


def test_bell_register_fidelity_sits_inside_the_intrinsic_budget(bell) -> None:  # type: ignore[no-untyped-def]
    """Section 9.6: the register state's infidelity against the compiled circuit's ideal state (its residual frame absorbed) lies
    below the closed-form budget reported beside the result (residual displacement, Debye-Waller, the (Omega/nu)^2 carrier scale,
    the addressing crosstalk) and is not zero: the physics is on."""
    fx, sur, res = bell
    fid = register_fidelity(res)
    budget = res.diagnostics.intrinsic_budget
    assert budget["total"] > 0.0 and "ms[2].residual_displacement" in budget
    assert 1e-5 < 1.0 - fid < budget["total"], (fid, budget)
    # the gate's own exact check bounds the register infidelity from below: the circuit adds five carrier pulses with crosstalk
    gate_inf = 1.0 - sur.entangling[(0, 1)].checks[-1].fidelity
    assert 1.0 - fid > 0.3 * gate_inf
    # against the uncompiled target the frame matters: (|00> + |11>)/sqrt2 differs from the played state by the frame's sign
    from tests.m6_fixtures import bell_target

    assert register_fidelity(res, bell_target()) < 0.05 or register_fidelity(res, bell_target()) > 0.95


def test_bell_diagnostics_report_the_space_classes_branches_and_approximations(bell) -> None:  # type: ignore[no-untyped-def]
    fx, sur, res = bell
    d = res.diagnostics
    # the caps follow the pulse's coherent excursion at the 1e-6 tail plus the 6-level margin (Section 5.5; M9a), and the margin
    # check never trips on them (no cap growth)
    assert d.level == "JOINT_EXACT" and d.space.dims[:2] == [2, 2] and len(d.space.dims) == 4
    assert all(10 <= x <= 14 for x in d.space.dims[2:]), d.space.dims
    assert d.cap_growth == {} and all(v >= 6 for v in d.margin_reached.values())
    assert d.dropped_contribution == (0.0, 0.0) and d.frozen_excitation_bound == {}
    assert d.mode_class == {
        0: "dropped",
        1: "dropped",
        2: "resolved",
        3: "resolved",
        4: "dropped",
        5: "dropped",
    }
    assert d.dropped_modes == (0, 1, 4, 5) and d.frozen_contribution == {}
    assert all(v < 1e-6 for v in d.boundary_population.values()) and set(d.boundary_population) == {2, 3}
    assert all(v >= 6 for v in d.margin_levels.values())
    assert d.samples == 1 and d.shots_per_sample == 2000 and d.effective_sample_size == 2000.0
    assert d.trajectories >= 3 and d.dropped_branch_weight < 1e-4
    assert d.integrator.startswith("dop853") and d.tolerances == (FAST.atol, FAST.rtol)
    assert d.root_seed == 0 and d.calibration is sur.table
    assert any(a.startswith("noise:") for a in d.approximations) and any(
        "product POVM" in a for a in d.approximations
    )
    assert d.wall_clock_span_s > 0.0
    assert res.spam["q0"][0] == pytest.approx(res.spam["q1"][0]) and 1e-4 < res.spam["q0"][0] < 5e-3
    assert res.spam["q0.state_preparation"][0] < 1e-4
    rec = last_record(res)
    assert (
        rec.compile.n_entangling == 1
        and len(rec.schedule.gates) == 1
        and rec.schedule.gates[0].pair == (0, 1)
    )
    meas = rec.schedule.measurement
    assert meas is not None and meas.ions == (0, 1)
    assert meas.t_end_s - meas.t_start_s == pytest.approx(sur.table.detection["window_s"].value)
    single = [p for p in rec.schedule.pulses if (p.gate_id or "").startswith("gpi2")]
    assert len(single) == 5 and all(abs(next(iter(p.drive.crosstalk.values()))) > 0.01 for p in single)
    assert rec.preparation.nbar[3] < 0.05 and rec.preparation.preparation_error(0) < 1e-4
    assert abs(rec.qubit_shifts_hz[0]) < 1e-9, "the surrogate frame sits at the derived transition"


def test_seeds_reproduce_shot_by_shot_and_readout_full_path_generates_records(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 3.4: the same root seed reproduces every shot, another seed draws different ones; the full readout path generates
    one photon record per ion per shot and agrees with the fast path within statistics."""
    fx, sur = two_ion
    kw = dict(
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    a = run(BELL, fx.device, 300, **kw)  # type: ignore[arg-type]
    b = run(BELL, fx.device, 300, **kw)  # type: ignore[arg-type]
    c = run(BELL, fx.device, 300, seed=3, **kw)  # type: ignore[arg-type]
    assert np.array_equal(a.bitstrings, b.bitstrings)
    assert not np.array_equal(a.bitstrings, c.bitstrings)
    assert a.diagnostics.dropped_branch_weight > 1e-3, "the coarse branch cutoff drops and reports weight"
    full = run(BELL, fx.device, 300, **kw, readout=Readout(mode="full"))  # type: ignore[arg-type]
    assert full.photon_records is not None and full.photon_records.shape == (300, 2)
    assert full.photon_records.max() > 3, "bright ions scatter tens of photons in the window"
    assert abs(full.probabilities.get("00", 0.0) - a.probabilities.get("00", 0.0)) < 0.1
    # the diagnostics say WHICH readout path ran (the `for s in ()` of the first version made this vacuous, M5 audit B13)
    assert any("readout full path" in s for s in full.diagnostics.approximations)
    assert any("photon record per ion per shot" in s for s in full.diagnostics.approximations)
    assert any("readout fast path" in s for s in a.diagnostics.approximations)
    assert not any("readout fast path" in s for s in full.diagnostics.approximations)


def test_single_qubit_gate_identity_on_the_pipeline(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 9.6 native-gate identities: GPi2(0) on |0> through the whole pipeline gives the Section 7.6 state within the
    carrier's off-resonant sideband scale plus the addressing crosstalk on the neighbour."""
    fx, sur = two_ion
    circ = Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1))
    res = run(
        circ,
        fx.device,
        100,
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-3)),
    )
    fid = register_fidelity(res)
    eps = sur.table.crosstalk[(0, 1)].value
    crosstalk_flip = math.sin(eps * math.pi / 4.0) ** 2
    assert 1.0 - fid < crosstalk_flip + 1e-3 and 1.0 - fid > 0.2 * crosstalk_flip
    assert res.diagnostics.space.resolved == () or all(
        m in (2, 3) for m in (t.mode for t in res.diagnostics.space.resolved)
    )
    assert res.diagnostics.mode_class[2] in ("frozen", "dropped"), (
        "no entangling gate: the x modes are not resolved"
    )


def test_refusals_and_the_level_guard(two_ion) -> None:  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    kw = dict(table=sur.table)
    with pytest.raises(ValueError, match="shots"):
        run(BELL, fx.device, 0, **kw)  # type: ignore[arg-type]
    mid = Circuit(2, (Operation("measure", (0,), ()), Operation("x", (0,), ())), (0, 1))
    with pytest.raises(ScheduleError, match="mid-circuit"):
        run(mid, fx.device, 10, **kw)  # type: ignore[arg-type]
    # a joint dimension above the guard routes to GATE_LOCAL (Section 11.5; M9a): the run proceeds and says so
    small_guard = run(
        BELL,
        fx.device,
        10,
        **kw,
        numerics=Numerics.from_solver_options(SolverOptions(joint_dimension_max=64)),
    )  # type: ignore[arg-type]
    assert small_guard.diagnostics.level == "GATE_LOCAL" and small_guard.diagnostics.gate_local is not None
    assert any("GATE_LOCAL" in a for a in small_guard.diagnostics.approximations)


def test_branch_enumeration_weights_and_cutoff() -> None:
    branches, dropped = enumerate_branches([[1.0 - 1e-3, 1e-3], [1.0, 0.0]], {2: 0.05, 3: 0.0}, 1e-4)
    weights = [b.weight for b in branches]
    assert weights == sorted(weights, reverse=True) and abs(sum(weights) + dropped - 1.0) < 1e-12
    assert branches[0].levels == (0, 0) and branches[0].fock == {2: 0, 3: 0}
    assert any(b.levels == (1, 0) for b in branches) and all(b.fock[3] == 0 for b in branches)
    assert 0.0 < dropped < 1e-3
    only, none = enumerate_branches([[1.0]], {}, 1e-6)
    assert len(only) == 1 and only[0].weight == 1.0 and none == 0.0


@pytest.mark.slow
def test_calibrate_and_run_without_a_table_build_the_surrogate_for_the_circuit_pairs() -> None:
    """Appendix E: run(table=None) calibrates the surrogate at t0 for the pairs the circuit uses; calibrate() is the same table."""
    fx = circuit_fixture(2)
    table = calibrate(
        Machine(fx.device), pairs=[(0, 1)], detection_records=1500, detection_windows_s=WINDOWS
    ).table
    assert table.waveform_for((0, 1)) is not None and table.detection["threshold"].status == "calibrated"
    # no table: run builds the closed-form surrogate for the circuit's pairs (the default scan settings)
    res = run(
        BELL, fx.device, 200, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2))
    )
    assert res.diagnostics.calibration.surrogate and res.probabilities.get("00", 0.0) > 0.3
    assert any("surrogate" in a or "waveform" in a for a in res.diagnostics.approximations)


def test_beat_phase_reset_offsets_the_legs_oppositely_and_keeps_the_spin_phase() -> None:
    """Section 7.10: hardware that programs each gate's tones from its own start resets the bichromatic beat note per gate; the
    red and blue legs shift by opposite phases 2 pi mu t_g, the spin phase (their half-sum) and hence the virtual-Z frame are
    unchanged, and phase-continuous hardware leaves the legs alone (Roos's tilt then belongs to the gate, M6 finding)."""
    import dataclasses

    from qutip_trap.control.schedule import beat_phase_offset_rad, schedule
    from qutip_trap.control.table import Waveform
    from tests.m4_fixtures import table_with_waveform, two_ion_modes

    fx = circuit_fixture(2, with_recipe=False)
    modes = two_ion_modes(fx.device)
    wf = Waveform.symmetric(modes, gate_mode=3, epsilon_hz=20e3)
    table = table_with_waveform((0, 1), wf, rabi_hz={(0, 2): 1e5, (1, 4): 1e5})
    circ = Circuit(
        2,
        (
            Operation("gpi2", (0,), (0.0,)),
            Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),
        ),
        (0, 1),
    )
    mu = float(wf.segments[0].detuning_hz["blue"])  # type: ignore[index]
    spin_by_mode: dict[bool, list[float]] = {}
    for continuous in (True, False):
        dev = dataclasses.replace(
            fx.device, hardware=dataclasses.replace(fx.device.hardware, phase_continuous=continuous)
        )
        sch = schedule(circ, dev, table)
        ms_pulses = [p for p in sch.pulses if (p.gate_id or "").startswith("ms")]
        t_g = ms_pulses[0].t_start_s
        assert t_g > 0.0
        offset = 0.0 if continuous else beat_phase_offset_rad(mu, t_g)
        for p in ms_pulses:
            red, blue = p.drive.tones
            assert (blue.phase_rad - red.phase_rad) % (2 * math.pi) == pytest.approx(
                (2 * offset) % (2 * math.pi), abs=1e-9
            )
        spin_by_mode[continuous] = [
            (0.5 * (p.drive.tones[0].phase_rad + p.drive.tones[1].phase_rad)) % math.pi for p in ms_pulses
        ]
    # the half-sum (the spin phase) is the same with and without the reset: the frame bookkeeping is untouched
    assert spin_by_mode[True] == pytest.approx(spin_by_mode[False], abs=1e-9)
    assert beat_phase_offset_rad(-mu, 3e-6) == pytest.approx(
        (-beat_phase_offset_rad(mu, 3e-6)) % (2 * math.pi), abs=1e-9
    )
    assert beat_phase_offset_rad(lambda tau: 1.0, 3.0) == 0.0, (
        "an FM leg is integrated from the pulse start by the builder"
    )


def _declared_probabilities(res, ions):  # type: ignore[no-untyped-def]
    """Exact declared-bit distribution over ``ions``: the register populations pushed through the readout POVM (no shot noise)."""
    import itertools

    rec = last_record(res)
    rho = res.final_state
    assert rho is not None
    n = res.n_qubits
    pop = np.real(np.diag(np.asarray(rho.full())))
    povm = rec.readout.product
    schemes = rec.readout.schemes
    out: dict[str, float] = {}
    for idx, p in enumerate(pop):
        if p < 1e-15:
            continue
        levels = [
            (idx >> (n - 1 - i)) & 1 for i in range(n)
        ]  # register order: ion 0 the most-significant index bit
        # the POVM takes the ions' true INTERNAL LEVELS, not their start classes (M5): identical for a two-level
        # ReadoutScheme.direct(1) whose bright level is 1, and wrong for any scheme whose bright level is 0
        for declared in itertools.product((True, False), repeat=n):
            q = povm.declared_bright_probability(levels, list(declared))
            if q <= 0.0:
                continue
            bits = [schemes[i].bit_of_class("bright" if declared[i] else "dark") for i in range(n)]
            key = "".join(str(bits[i]) for i in reversed(ions))
            out[key] = out.get(key, 0.0) + float(p) * float(q)
    return out


@pytest.mark.slow
def test_three_ion_ghz_circuit_resolves_two_modes_and_freezes_the_tilt() -> None:
    """Section 9.6 row 2 through run(): H, CNOT(0,1), CNOT(1,2) on the three-ion chain; the adjacent pairs resolve the COM and
    zigzag modes and freeze the tilt (whose participation on the middle ion vanishes), the frozen contribution is reported, the
    boundary populations stay below the threshold and the histogram is the GHZ one within the readout and the crosstalk."""
    fx = circuit_fixture(3, address_waist_m=2.0e-6)
    sur = surrogate_table(
        fx.device, pairs=[(0, 1), (1, 2)], detection_records=1500, detection_windows_s=WINDOWS
    )
    assert 0.01 < sur.table.crosstalk[(1, 0)].value < 0.04
    for pair in ((0, 1), (1, 2)):
        # Section 5.2 with the M9 third drop condition: the tilt's LOOP pair (|alpha|^2 (2n+1) = 3.5e-33, chi = 0) sits below
        # (1e-6, 1e-4) because the AM pulse closes its loop exactly, but its eta = 0.0809 at the prepared nbar = 0.0214
        # carries a Debye-Waller spread eta^2 sqrt(nbar(nbar+1)) = 9.69e-4 rad, 3.2x above DW_SPREAD_DROP_MAX = 3e-4, so it is
        # FROZEN and the per-shot draw of Section 5.2 carries the spread the calibration cannot absorb
        # (conv.drop_test_debye_waller_spread; before that condition it was dropped and the factor left the dynamics)
        assert sur.mode_classes[pair][4] == "frozen" and sur.mode_classes[pair][5] == "resolved"
        check = sur.entangling[pair].checks[-1]
        wf = sur.table.waveform_for(pair)
        assert wf is not None and wf.segments is not None
        peak = max(abs(float(seg.amplitude_hz[(pair[0], "blue")])) for seg in wf.segments)
        carrier_scale = (2.0 * math.pi * peak / (2.0 * math.pi * 2.569e6)) ** 2
        # the seven-segment pulse on three modes is a strong drive: its leakage sits inside the (Omega/nu)^2 carrier scale
        assert check.fidelity > 0.99 and check.leakage < carrier_scale, (
            check.fidelity,
            check.leakage,
            carrier_scale,
        )
    ghz = Circuit(
        3, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()), Operation("cnot", (1, 2), ())), (0, 1, 2)
    )
    res = run(
        ghz,
        fx.device,
        1000,
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=3e-3)),
    )
    d = res.diagnostics
    # two resolved modes: the cap rule of Section 5.5 at the 1e-6 tail (M9a) gives 11 levels on the COM (populated to n = 2) and
    # 13 on the zigzag (populated to n = 4, the larger excursion), each with the Section 5.1.1 margin; the M6 rule's 10 to 12
    # range this assertion carried until M9b was stale from the M9a rule change (the test failed identically at the M9a commit)
    assert (
        d.space.dims[:3] == [2, 2, 2]
        and len(d.space.dims) == 5
        and all(10 <= x <= 13 for x in d.space.dims[3:])
        and d.mode_class[4] == "frozen"
        and set(d.frozen_contribution) == {4}
    )
    # the tilt is frozen by the Debye-Waller-spread condition (see the comment above), so its chi_m loss and its off-resonant
    # excitation are reported and the factor stays in the dynamics; nothing is DROPPED among the modes an entangling gate
    # touches on this fixture, and the modes that are dropped (the y and z families, eta = 0 for a Delta k along x) have no
    # contribution to report, so Section 11.3 item 2's summed dropped contribution is legitimately zero here
    assert 4 not in d.dropped_modes and d.dropped_contribution == (0.0, 0.0)
    assert d.frozen_contribution[4][0] < 1e-6 and d.frozen_contribution[4][1] < 1e-4
    assert d.frozen_excitation_bound.get(4, 0.0) > 0.0
    assert all(v < 1e-6 for v in d.boundary_population.values())
    p = res.probabilities
    assert p.get("000", 0.0) + p.get("111", 0.0) > 0.95
    assert abs(p.get("000", 0.0) - p.get("111", 0.0)) < 6.0 * res.error_bars["000"]
    budget = d.intrinsic_budget
    assert any(k.endswith(".crosstalk") for k in budget), (
        "the single-qubit pulses' addressing crosstalk is budgeted"
    )
    assert 1.0 - register_fidelity(res) < budget["total"]
    assert register_fidelity(res) > 0.97


@pytest.mark.slow
def test_bernstein_vazirani_errors_emerge_predominantly_as_one_to_zero_flips() -> None:
    """Section 9.6 row 6 (Wright's minimal model as a test of what emerges): Bernstein-Vazirani with the secret 01 on data ions 0
    and 2, the middle ion the ancilla, 1.3 % addressing crosstalk; the exact declared distribution (register state through the
    POVM) puts more weight on the 1 -> 0 flip of the secret than on the 0 -> 1 flip: the oracle CNOT maps the ancilla's
    crosstalk rotations onto its control (the secret's 1 bit) while the 0 bit sees only the ancilla pulses' direct rotations,
    and the bright-state readout error exceeding the dark one adds to it (check_circuits.py 4 prints the coherent-only
    matrix model beside the exact register populations)."""
    fx = circuit_fixture(3, address_waist_m=2.0e-6)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=1500, detection_windows_s=WINDOWS)
    ops = (
        Operation("x", (1,), ()),
        Operation("h", (0,), ()),
        Operation("h", (1,), ()),
        Operation("h", (2,), ()),
        Operation("cnot", (0, 1), ()),
        Operation("h", (0,), ()),
        Operation("h", (1,), ()),
        Operation("h", (2,), ()),
    )
    bv = Circuit(3, ops, (0, 1, 2))
    ideal = ideal_probabilities(Circuit(3, ops, (0, 2)))
    assert ideal == pytest.approx({"01": 1.0})
    res = run(
        bv,
        fx.device,
        1000,
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=3e-3)),
    )
    declared = _declared_probabilities(res, (0, 2))
    assert declared["01"] > 0.99
    one_to_zero = declared.get("00", 0.0)
    zero_to_one = declared.get("11", 0.0)
    assert one_to_zero > zero_to_one > 0.0, declared
    assert 1e-5 < one_to_zero + zero_to_one < 3e-2
    assert 1.0 - register_fidelity(res) < res.diagnostics.intrinsic_budget["total"]
    # the sampled histogram over the measured data ions agrees with the exact declared distribution
    hist = {}
    for row in res.bitstrings:
        key = f"{row[2]}{row[0]}"
        hist[key] = hist.get(key, 0) + 1
    assert hist.get("01", 0) / res.shots == pytest.approx(declared["01"], abs=0.02)
