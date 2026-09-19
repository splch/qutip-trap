"""run() with the noise layer on (PLAN.md Sections 3.4, 6.1, 6.6, 6.7, 7.5, 7.10; Section 9.17 rows 'Shot clock', 'Seeds and
reproducibility'; M7): dynamical samples at the shot clock, the effective sample size, the trajectory path, leakage levels,
collisions with heralds and the persistent machine state, and the scheduler's crosstalk-suppression echoes."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.api import (
    Circuit,
    Collisions,
    Drift,
    Operation,
    SolverOptions,
    last_record,
    register_fidelity,
    run,
    schedule,
    white_spectrum,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import compile_report
from qutip_trap.options import Numerics, Physics
from qutip_trap.run.job import effective_sample_size
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
TORR_PA = 133.32236842105263


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=1000, detection_windows_s=WINDOWS)
    return fx, sur


def _noisy(device, **fields):  # type: ignore[no-untyped-def]
    return dataclasses.replace(device, noise=dataclasses.replace(device.noise, **fields))


def test_effective_sample_size_between_and_within_samples() -> None:
    """Var(p_hat) = Var_total(p_s)/S: i.i.d. samples must return the full shot count (adding the within-sample binomial term
    on top of the total per-sample variance halves n_eff, which the bands below are tight enough to catch)."""
    rng = np.random.default_rng(0)
    same = [rng.integers(0, 2, size=(100, 1)).astype(np.uint8) for _ in range(4)]
    n_same = effective_sample_size(same)
    assert 0.9 * 400.0 < n_same <= 400.0 + 1e-9
    # 16 samples concentrate the variance estimate, so the double-counted form lands near 400 and this band excludes it
    rng = np.random.default_rng(0)
    many = [rng.integers(0, 2, size=(50, 1)).astype(np.uint8) for _ in range(16)]
    n_many = effective_sample_size(many)
    assert 0.9 * 800.0 < n_many <= 800.0 + 1e-9
    # a deliberate between-sample spread (quasi-static drift): the samples are far from independent shots
    rng = np.random.default_rng(7)
    spread = [
        (rng.random((100, 1)) < p).astype(np.uint8) for p in (0.2, 0.35, 0.5, 0.65, 0.8, 0.45, 0.55, 0.3)
    ]
    n_spread = effective_sample_size(spread)
    assert n_spread < 0.1 * 800.0
    # sigma_between = 0.19 over the eight means, so Var(p_s)/S = 0.19^2/8 and n_eff ~ p(1-p) 8/0.19^2 ~ 55
    assert 40.0 < n_spread < 80.0
    biased = [np.zeros((100, 1), np.uint8), np.ones((100, 1), np.uint8)]
    assert effective_sample_size(biased) < 10.0
    assert effective_sample_size([np.zeros((50, 2), np.uint8)]) == 50.0


def test_effective_sample_size_weights_unequal_sample_sizes() -> None:
    """Unequal M_s: the weighted total-variance estimator reduces to var(p_s, ddof=1)/S when the counts are equal and stays
    an unbiased Var(p_hat) estimate when they are not (a 900/100 split of i.i.d. shots is still 1000 independent shots)."""
    rng = np.random.default_rng(3)
    uneven = [
        rng.integers(0, 2, size=(n, 1)).astype(np.uint8) for n in (900, 100, 500, 500, 250, 250, 250, 250)
    ]
    n_uneven = effective_sample_size(uneven)
    assert 0.5 * 3000.0 < n_uneven <= 3000.0 + 1e-9


@pytest.mark.slow
def test_quasi_static_drift_gives_several_samples_and_a_reduced_effective_sample_size(two_ion) -> None:  # type: ignore[no-untyped-def]
    """A field drift and a Rabi drift sampled at the shot clock: two samples carry different offsets, the shots split into contiguous blocks at the shot clock (conv.shot_blocks_per_sample),
    the error bars use n_eff <= shots, and the same seed reproduces everything (Section 3.4)."""
    fx, sur = two_ion
    dev = _noisy(fx.device, field_drift=Drift(4e-7, 10.0, None), rabi_drift=Drift(2e-2, 1.0, None))
    kw = dict(
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2), samples=2),
        keep_final_state=True,
    )
    res = run(BELL, dev, 300, **kw)  # type: ignore[arg-type]
    d = res.diagnostics
    assert d.samples == 2 and d.shots_per_sample == 150 and len(res.noise_samples) == 2
    assert res.noise_samples[1].t_s == pytest.approx(150 * d.wall_clock_span_s / 299.0, rel=1e-9)
    offs = [s.values["qubit_offset_hz[0]"] for s in res.noise_samples]
    assert offs[0] != offs[1] and all(abs(o) > 0.01 for o in offs)
    assert (
        1.0 <= d.effective_sample_size <= 300.0
        and res.error_bars["00"]
        >= math.sqrt(res.probabilities["00"] * (1 - res.probabilities["00"]) / 300.0) * 0.999
    )
    assert any("dynamical samples" in a for a in d.approximations)
    again = run(BELL, dev, 300, **kw)  # type: ignore[arg-type]
    assert np.array_equal(again.bitstrings, res.bitstrings)
    assert res.probabilities["00"] + res.probabilities["11"] > 0.97


@pytest.mark.slow
def test_heating_channels_route_to_trajectories_above_the_mesolve_dimension(two_ion) -> None:  # type: ignore[no-untyped-def]
    """S_E with an uncorrelated correlation length assembles heating channels on the resolved modes; at dimension 400 the run takes
    the keyed trajectory path (ntraj per branch), reports the method, and the register state stays close to the quiet one."""
    fx, sur = two_ion
    dev = _noisy(fx.device, S_E=white_spectrum(1e-13, "(V/m)^2/(rad/s)"), correlation_length_m=0.0)
    assert dev.noise.heating_rates_quanta_per_s(dev)[3] > 1.0
    kw = dict(
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2, ntraj=3)),
        keep_final_state=True,
    )
    res = run(BELL, dev, 200, **kw)  # type: ignore[arg-type]
    d = res.diagnostics
    assert d.trajectories >= 3 and any(a.startswith("noise:") and "mcsolve" in a for a in d.approximations)
    quiet = run(BELL, fx.device, 200, **{**kw, "physics": Physics(noise=False)})  # type: ignore[arg-type]
    assert abs(register_fidelity(res) - register_fidelity(quiet)) < 5e-3
    rec = last_record(res)
    assert all(t.final.joint is not None for t in rec.traces)


# eight serial trajectories at dimension 1287: four minutes here, past thirty on the loaded 4-vCPU runner, so an hour of its own
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_leakage_levels_extend_the_register_and_the_readout_classes(two_ion) -> None:  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    # internal_levels = 3 turns the scattering channels on automatically (Section 4.5.5; conv.scattering_channels_at_d_gt_2):
    # 28 register-only leakage, spin-flip and Rayleigh operators per pulse ride along on the trajectory path (dimension
    # 1287 is above mesolve_dimension_max). Eight keyed trajectories per branch keep the test at about three minutes where
    # the default 64 made it a 25-minute one; the assertions below are coarse enough for that count (measured F = 0.9976)
    kw = dict(
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2, ntraj=8)),
        keep_final_state=True,
        physics=Physics(internal_levels=3),
    )
    res = run(BELL, fx.device, 200, **kw)  # type: ignore[arg-type]
    d = res.diagnostics
    assert d.space.ion_dims == (3, 3) and res.final_state is not None and res.final_state.dims[0] == [3, 3]
    assert any("leakage levels" in a and "SINK" in a for a in d.approximations)
    assert any("scattering_channels turned ON with scattering_recoil='off'" in a for a in d.approximations)
    assert register_fidelity(res) > 0.99
    rec = last_record(res)
    assert all(s.n_levels == 3 and s.classes[2] == "dark" for s in rec.readout.schemes)
    assert res.probabilities["00"] + res.probabilities["11"] > 0.97


@pytest.mark.slow
def test_collisions_herald_and_discard_shots_and_flag_ions(two_ion) -> None:  # type: ignore[no-untyped-def]
    """An absurd pressure makes collisions frequent: heating kicks during the cooling stage are heralded and kept with their drawn
    energy reported (Section 6.7's k_B T m_gas/m_ion scale), events during the sequence discard the shot, a reorder permutes
    RunState.order from the configured permutation distribution, and a loss or dark-ion event flags the ion so that every later shot
    reads it dark (Section 6.7). All four outcomes are enabled here: the audit found reorder and loss pinned to zero, so neither
    branch ever executed in CI."""
    fx, sur = two_ion
    col = Collisions(
        3e-6 * TORR_PA,
        {"H2": 1.0},
        {"heating_kick": 0.4, "reorder": 0.3, "loss": 0.15, "dark_ion": 0.15},
        reorder_permutations=((1, 0),),
    )
    dev = _noisy(fx.device, collisions=col)
    kw = dict(
        table=sur.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    res = run(BELL, dev, 120, **kw)  # type: ignore[arg-type]
    d = res.diagnostics
    assert res.discarded_shots > 0 and res.shots + res.discarded_shots == 120
    assert res.heralds.shape == (res.shots,) and int(np.sum(res.heralds & 1)) > 0
    labels = [e[1] for e in res.run_state.events]
    assert any(lab.startswith("collision:") for lab in labels)
    outcomes = {lab.split(":")[1] for lab in labels if lab.startswith("collision:")}
    assert len(outcomes) >= 3, f"at this pressure every branch should fire: {sorted(outcomes)}"
    if "heating_kick" in outcomes:
        # the drawn kick is reported in quanta of the softest mode: Section 6.7's "tens to thousands"
        kicks = [float(lab.split("dnbar=")[1]) for lab in labels if "dnbar=" in lab]
        assert kicks and all(q >= 0.0 for q in kicks) and max(kicks) > 1.0
        assert any(
            "heating kick(s) drawn from the Section 6.7 energy distribution" in a for a in d.approximations
        )
    if "reorder" in outcomes:
        # every reorder applies the configured (1, 0) transposition to the PERSISTENT order, so the final order is the
        # identity exactly when an even number of them fired
        n_reorder = sum(1 for lab in labels if lab.startswith("collision:reorder"))
        expected = tuple(range(2)) if n_reorder % 2 == 0 else (1, 0)
        assert res.run_state.order == expected, (n_reorder, res.run_state.order)
        assert any("reorder event(s) permuted RunState.order" in a for a in d.approximations)
        assert any("does not re-derive b_{i,m}" in a for a in d.approximations), (
            "the declared approximation: a reorder does not change the mode structure of later shots"
        )
    if res.run_state.dark or res.run_state.lost:
        assert int(np.sum(res.heralds & 2)) > 0
        assert any("nominal crystal" in a for a in d.approximations)
    quiet = run(BELL, fx.device, 120, **kw)  # type: ignore[arg-type]
    assert quiet.discarded_shots == 0 and not quiet.run_state.dark and quiet.heralds.sum() == 0
    assert quiet.run_state.order == tuple(range(2))


def test_crosstalk_suppression_schedules_the_echoes_of_section_6_6() -> None:
    fx = circuit_fixture(3)
    sur = (
        surrogate_table(
            fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=(20e-6,), spot_check=False
        )
        if "spot_check" in surrogate_table.__code__.co_varnames
        else surrogate_table(fx.device, pairs=[(0, 1)], detection_records=200, detection_windows_s=(20e-6,))
    )
    ms_circ = Circuit(3, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1, 2))
    rep = compile_report(ms_circ, fx.device)
    plain = schedule(rep.circuit, fx.device, sur.table)
    local = schedule(rep.circuit, fx.device, sur.table, crosstalk_suppression="local")
    neigh = schedule(rep.circuit, fx.device, sur.table, crosstalk_suppression="neighbour")
    assert len(plain.gates) == 1 and len(local.gates) == 2 and len(neigh.gates) == 2
    assert all(
        abs(g.waveform.chi_total_rad)
        == pytest.approx(abs(plain.gates[0].waveform.chi_total_rad) / 2.0, rel=1e-9)
        for g in local.gates
    )
    echo_local = [p for p in local.pulses if "/echo/" in (p.gate_id or "") or "/unecho/" in (p.gate_id or "")]
    assert len(echo_local) == 4 and {p.drive.ions[0] for p in echo_local} == {0, 1}
    echo_neigh = [p for p in neigh.pulses if "/echo/" in (p.gate_id or "")]
    assert len(echo_neigh) == 2 and {p.drive.ions[0] for p in echo_neigh} == {2}, (
        "the spectator gets X(pi) then Y(pi) = Z(pi)"
    )
    # Z(pi) plus the virtual-Z frame the two compensated echo pulses leave behind (2 pi delta_St t_pi each, M8 Section 7.5 item 7)
    from qutip_trap.control.schedule import stark_phase_rad

    stark_frame = sum(stark_phase_rad(p) for p in echo_neigh)
    assert stark_frame != 0.0 and abs(stark_frame) < 0.01
    assert neigh.phase_frame[2] == pytest.approx(math.pi + stark_frame) and local.phase_frame[2] == 0.0
    assert local.duration_s > plain.duration_s and neigh.duration_s > plain.duration_s
