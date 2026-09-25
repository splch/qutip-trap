"""End-to-end circuits in JOINT_EXACT (PLAN.md Sections 3.4, 5.2, 5.7, 7.2, 7.3, 8.6, 9.6): the surrogate table, the Bell
state with all physics on against the ideal distribution and the intrinsic budget, the diagnostics, seeds, the two
readout paths and every discriminator of Section 8.3, circuits on fewer qubits than ions, and the three- and four-ion rows
(GHZ, Wright's crosstalk model, row 2b) as slow tests."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qutip_trap.calibration import calibrate
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation, ideal_probabilities
from qutip_trap.control.schedule import ScheduleError
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.readout.discriminate import AdaptiveML, FirstPhoton, ThresholdDiscriminator, TimeResolvedML
from qutip_trap.run.job import enumerate_branches, ideal_register_state, last_record, register_fidelity
from tests.fixtures import run
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
ONE = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
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
    # the 171Yb+ Raman intermediate sum over P1/2 and P3/2: the two paths add for this polarization (147 kHz)
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
    """Section 9.6: the two-ion Bell state with all physics on; P_00 ~ P_11, the odd strings at the readout-error level."""
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
    """Section 9.6: the register infidelity against the compiled circuit's ideal state (its residual frame absorbed) lies
    below the closed-form budget reported beside the result and is not zero: the physics is on."""
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
    # the caps follow the pulse's coherent excursion at the 1e-6 tail plus the 6-level margin (Section 5.5), and the margin
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
    # the diagnostics say which readout path ran
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
    # a joint dimension above the guard routes to GATE_LOCAL (Section 11.5): the run proceeds and says so
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


def test_beat_phase_reset_offsets_the_legs_oppositely_and_keeps_the_spin_phase() -> None:
    """Section 7.10: hardware that programs each gate's tones from its own start resets the bichromatic beat note per gate;
    the red and blue legs shift by opposite phases 2 pi mu t_g, the spin phase (their half-sum) and hence the virtual-Z frame
    are unchanged, and phase-continuous hardware leaves the legs alone."""
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


# ---- circuits on fewer qubits than ions -------------------------------------------------------------------------------------


def _fast_run(two_ion, circuit: Circuit, shots: int = 40):  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    opts = SolverOptions(branch_weight_min=1e-3)
    return run(
        circuit,
        fx.device,
        shots,
        table=sur.table,
        keep_final_state=True,
        physics=Physics.from_solver_options(opts),
        numerics=Numerics.from_solver_options(opts),
    )


def test_register_fidelity_of_a_circuit_on_fewer_qubits_than_ions(two_ion) -> None:  # type: ignore[no-untyped-def]
    """A one-qubit circuit on the two-ion chain: the register state is 4 x 4, the ideal ket is |+> on ion 0 and |0> on the
    idle ion 1, and the fidelity is that of the played pi/2 pulse."""
    res = _fast_run(two_ion, Circuit(1, (Operation("h", (0,), ()),), (0,)))
    assert res.final_state is not None and res.final_state.dims[0] == [2, 2]
    assert res.n_qubits == 1 and res.bitstrings.shape == (40, 1)
    fid = register_fidelity(res)
    assert 0.99 < fid <= 1.0 + 1e-12
    # the same number by hand: the compiled circuit's ket on ion 0 (its frame absorbed) with |0> on the idle ion 1
    rho = np.asarray(res.final_state.full())
    ideal = np.kron(ideal_register_state(res), np.array([1.0, 0.0])).astype(complex)
    assert fid == pytest.approx(float(np.real(ideal.conj() @ rho @ ideal)), abs=1e-12)
    # an explicit target on the circuit's qubits is embedded the same way
    plus = np.array([1.0, 1.0]) / math.sqrt(2.0)
    plus0 = np.kron(plus, np.array([1.0, 0.0])).astype(complex)
    assert register_fidelity(res, plus) == pytest.approx(
        float(np.real(plus0.conj() @ rho @ plus0)), abs=1e-12
    )
    with pytest.raises(ValueError, match="spans 3 qubits"):
        register_fidelity(res, np.ones(8) / math.sqrt(8.0))


def test_register_fidelity_when_the_circuit_measures_a_subset(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The Bell circuit measuring qubit 0 only: one histogram column, the fidelity against the full two-qubit Bell ket."""
    bell_q0 = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0,))
    res = _fast_run(two_ion, bell_q0)
    assert res.n_qubits == 1 and res.qubits == (0,) and set(res.counts) <= {"0", "1"}
    fid = register_fidelity(res)
    assert 0.98 < fid <= 1.0 + 1e-12


def test_run_record_outcome_covers_every_kept_shot(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The readout runs once per (sample, branch) batch; the RunRecord's outcome is the concatenation over the kept shots in
    the Result's row order, over every ion, so a per-shot reader lines up with ``Result.bitstrings``."""
    res = _fast_run(two_ion, BELL, shots=60)
    rec = last_record(res)
    assert len(rec.branches) > 1, (
        "the fixture's initial mixture has several branches, so several readout batches"
    )
    out = rec.outcome
    assert out.bits.shape == (res.shots, 2) and out.levels.shape == (res.shots, 2)
    assert out.time_used_s.shape == (res.shots, 2)
    assert np.array_equal(out.bits, res.bitstrings), "every ion measured: the declared bits are the Result's"
    assert out.mode == "fast" and out.records is None


# ---- every discriminator of Section 8.3 through the run -------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(1)
    sur = surrogate_table(
        fx.device,
        pairs=[],
        detection_records=400,
        detection_windows_s=tuple(float(x) for x in np.linspace(10e-6, 40e-6, 4)),
    )
    kw = {
        "table": sur.table,
        "numerics": Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
        "physics": Physics(noise=False),
    }
    return fx, kw


def test_time_resolved_ml_runs_end_to_end_on_the_full_record_path(one_ion) -> None:  # type: ignore[no-untyped-def]
    """The record path replaces the POVM (Section 5.7), so no Monte-Carlo confusion is built; the sub-bin record and the
    posterior confidence of Section 8.6 reach ``Result``."""
    fx, kw = one_ion
    disc = TimeResolvedML(5e-6, 20e-6)
    res = run(ONE, fx.device, 40, **kw, readout=Readout(mode="full", discriminator=disc))
    rec = last_record(res)
    assert rec.readout.povm is None and rec.readout.product is None
    # the scheme states the dark class: 171Yb+ direct fluorescence, so "dark" and Myerson's 1/tau is R_b
    assert rec.readout.discriminator.dark_class == "dark"  # type: ignore[union-attr]
    assert res.photon_records is not None and res.photon_records.shape == (40, 1)
    assert res.sub_bin_records is not None and res.sub_bin_records.shape == (40, 1, 4)
    assert np.array_equal(res.sub_bin_records.sum(axis=2), res.photon_records)
    assert res.posteriors is not None and res.posteriors.shape == (40, 1)
    assert np.all(res.posteriors >= 0.0) and np.all(res.posteriors <= 1.0)
    assert any("SPAM definition" in s and "TimeResolvedML" in s for s in res.diagnostics.approximations)
    assert any("estimated from this run's own" in s for s in res.diagnostics.approximations)
    # a level the circuit never populates reports nan rather than a silent zero
    eps = res.spam["q0"]
    assert all(math.isnan(x) or 0.0 <= x <= 1.0 for x in eps)
    assert sum(res.probabilities.values()) == pytest.approx(1.0)


def test_a_monte_carlo_povm_runs_the_fast_path_and_declares_its_error_bar(one_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 8.4: a non-threshold discriminator's confusion has no closed form, so the fast path estimates it and reports
    the statistical uncertainty it earned (``POVM.uncertainty``)."""
    fx, kw = one_ion
    disc = TimeResolvedML(5e-6, 20e-6)
    res = run(ONE, fx.device, 40, **kw, readout=Readout(discriminator=disc, povm_samples=400))
    rec = last_record(res)
    assert rec.readout.povm is not None and rec.readout.povm.per_ion is not None
    unc = rec.readout.povm.uncertainty
    assert unc == pytest.approx(math.sqrt(0.25 / 400), rel=1e-12)
    assert any(f"{unc:.2e}" in s for s in res.diagnostics.approximations)
    assert any("400 sampled" in s for s in res.diagnostics.approximations)
    eps_b, eps_d = res.spam["q0"]
    assert 0.0 <= eps_b <= 0.1 and 0.0 <= eps_d <= 0.1
    # the threshold discriminator's POVM stays exact and reports no uncertainty
    thr = run(ONE, fx.device, 40, **kw, readout=Readout(discriminator=ThresholdDiscriminator(0.5, 20e-6)))
    assert last_record(thr).readout.povm.uncertainty == 0.0  # type: ignore[union-attr]
    assert not any("sampled records per level" in s for s in thr.diagnostics.approximations)


def test_the_adaptive_and_first_photon_protocols_also_run(one_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 8.3's adaptive Bayesian early termination and Noek's / Crain's first-photon protocols; the first-photon
    record needs arrival times, which Section 8.6 asks ``Result`` to keep."""
    fx, kw = one_ion
    for disc in (
        AdaptiveML(5e-6, 40e-6, 1e-3),
        FirstPhoton(40e-6, cutoff_s=10e-6),
    ):
        res = run(ONE, fx.device, 30, **kw, readout=Readout(mode="full", discriminator=disc))
        assert res.shots == 30 and res.photon_records is not None
        assert any("readout full path" in s for s in res.diagnostics.approximations)
        if disc.needs_arrivals:
            assert res.arrival_times_s is not None and len(res.arrival_times_s) == 30
            assert all(len(shot) == 1 for shot in res.arrival_times_s)
            assert all(
                len(shot[0]) == int(total)
                for shot, total in zip(res.arrival_times_s, res.photon_records[:, 0])
            )
        else:
            assert res.arrival_times_s is None


# ---- three- and four-ion rows of Section 9.6 --------------------------------------------------------------------------------


@pytest.mark.slow
def test_calibrate_and_run_without_a_table_build_the_surrogate_for_the_circuit_pairs() -> None:
    """A run without a table calibrates the surrogate for the pairs the circuit uses; ``calibrate`` is the same table."""
    fx = circuit_fixture(2)
    table = calibrate(
        Machine(fx.device), pairs=[(0, 1)], detection_records=1500, detection_windows_s=WINDOWS
    ).table
    assert table.waveform_for((0, 1)) is not None and table.detection["threshold"].status == "calibrated"
    res = run(
        BELL, fx.device, 200, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2))
    )
    assert res.diagnostics.calibration.surrogate and res.probabilities.get("00", 0.0) > 0.3
    assert any("surrogate" in a or "waveform" in a for a in res.diagnostics.approximations)


def _declared_probabilities(res, ions):  # type: ignore[no-untyped-def]
    """Exact declared-bit distribution over ``ions``: the register populations pushed through the readout POVM (no shot noise)."""
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
        # the POVM takes the ions' true INTERNAL LEVELS, not their start classes
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
    """Section 9.6 row 2: H, CNOT(0,1), CNOT(1,2) on the three-ion chain; the adjacent pairs resolve the COM and zigzag modes
    and freeze the tilt (whose participation on the middle ion vanishes), the frozen contribution is reported, the boundary
    populations stay below the threshold and the histogram is the GHZ one within the readout and the crosstalk."""
    fx = circuit_fixture(3, address_waist_m=2.0e-6)
    sur = surrogate_table(
        fx.device, pairs=[(0, 1), (1, 2)], detection_records=1500, detection_windows_s=WINDOWS
    )
    assert 0.01 < sur.table.crosstalk[(1, 0)].value < 0.04
    for pair in ((0, 1), (1, 2)):
        # the tilt's loop pair sits below the drop pair (the AM pulse closes its loop exactly), but its Debye-Waller spread
        # eta^2 sqrt(nbar(nbar+1)) = 9.69e-4 rad is above DW_SPREAD_DROP_MAX = 3e-4, so it is FROZEN
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
    # two resolved modes, each capped by the Section 5.5 rule at the 1e-6 tail plus the Section 5.1.1 margin
    assert (
        d.space.dims[:3] == [2, 2, 2]
        and len(d.space.dims) == 5
        and all(10 <= x <= 13 for x in d.space.dims[3:])
        and d.mode_class[4] == "frozen"
        and set(d.frozen_contribution) == {4}
    )
    # the frozen tilt's chi_m loss and off-resonant excitation are reported; the dropped modes (the y and z families,
    # eta = 0 for a Delta k along x) contribute nothing
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
    and 2, the middle ion the ancilla, 1.3 % addressing crosstalk; the exact declared distribution puts more weight on the
    1 -> 0 flip of the secret than on the 0 -> 1 flip: the oracle CNOT maps the ancilla's crosstalk rotations onto its
    control while the 0 bit sees only the ancilla pulses' direct rotations, and the bright-state readout error exceeding the
    dark one adds to it."""
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


@pytest.fixture(scope="module")
def four_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(4, address_waist_m=2.0e-6)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=800, detection_windows_s=WINDOWS)
    return fx, sur


@pytest.mark.slow
def test_a_four_ion_circuit_runs_through_the_pipeline_at_the_row_2b_dimension(four_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 9.6 row 2b, "four ions with two resolved modes at d_m = 12 (dimension 2304)", on an explicit space (the
    pipeline's own selection resolves three x modes and is above the guard): a Bell pair on ions 0 and 1, all physics on,
    the two spectators dark, the truncation monitor inside its threshold."""
    fx, sur = four_ion
    resolved = (6, 7)  # the x-COM at 3.0 MHz and the tilt at 2.8284 MHz
    space = HilbertSpace(
        (2, 2, 2, 2),
        tuple(ModeTruncation(m, 12, (0, 4), 0.15) for m in resolved),
        None,
        tuple(m for m in range(12) if m not in resolved),
    )
    circuit = Circuit(4, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1, 2, 3))
    result = run(
        circuit,
        fx.device,
        200,
        table=sur.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=3e-3), space=space),
    )
    d = result.diagnostics
    assert d.level == "JOINT_EXACT"
    assert d.space.dimension == 2304 and d.space.dims == [2, 2, 2, 2, 12, 12]
    assert tuple(t.mode for t in d.space.resolved) == resolved and all(t.d == 12 for t in d.space.resolved)
    assert all(v < 1e-6 for v in d.boundary_population.values())
    assert d.cap_growth == {}
    # ions 2 and 3 were never addressed, so every shot reads them dark: the histogram lives on 00xx and 11xx
    probabilities = result.probabilities
    assert probabilities.get("0000", 0.0) + probabilities.get("0011", 0.0) > 0.9, probabilities
    assert 1.0 - register_fidelity(result) < d.intrinsic_budget["total"]
    assert result.bitstrings.shape == (200, 4)


@pytest.mark.slow
def test_the_pipeline_s_own_four_ion_space_exceeds_the_guard_and_routes_to_gate_local(four_ion) -> None:  # type: ignore[no-untyped-def]
    """The adjacent-pair waveform resolves three of the four x modes, whose caps put the joint space above the Section 11.5
    guard, so the run takes the GATE_LOCAL level and says so (Section 5.4)."""
    fx, sur = four_ion
    circuit = Circuit(4, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1, 2, 3))
    result = run(
        circuit,
        fx.device,
        50,
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    d = result.diagnostics
    assert d.level == "GATE_LOCAL" and d.gate_local is not None
    assert any("GATE_LOCAL" in note for note in d.approximations)
    resolved = sorted(t.mode for t in d.space.resolved)
    assert len(resolved) == 3 and set(resolved) <= {4, 5, 6, 7}, resolved
    assert d.space.dimension > 4096
