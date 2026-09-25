"""The engine's exact shortcuts (constant segments, the rotating-frame idle, pure branches) against independent references,
the tolerance check and the integrator ladder, and the Fock marginals and wall times of ``Traces`` (PLAN.md Section 5.3)."""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest
import qutip as qt

import qutip_trap as trap
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, _pure_branches
from qutip_trap.dynamics.evolve import (
    LARGE_MODE_ATOL,
    LARGE_MODE_DIMENSION,
    convergence_check,
    evolve,
    tightened,
)
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian, interaction_picture
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.dynamics.truncation import regrid_state
from qutip_trap.experiments.single_ion import ramsey
from qutip_trap.light.microwave import square_microwave_drive
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.machine import Machine
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics
from qutip_trap.run.job import last_record
from qutip_trap.units import TWO_PI
from tests.fixtures import BELL, KX, single_ion_raman_device, two_ion_raman_device
from tests.oracles import sideband_rabi_rad_s

WX = TWO_PI * 3.0e6
"""The references integrate the schedule as written, so the engine does not play it through the hardware chain."""


@pytest.fixture(scope="module")
def raman():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 6), 0.2),), None, (0, 2))
    return dev, dd, space


def _run(dev, sched, state, space, opts=None, **engine_kw):  # type: ignore[no-untyped-def]
    eng = JointExactEngine(store_per_segment=4, hardware_chain=False, **engine_kw)
    return eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(0), opts or Numerics()), eng


def _heated(dev):  # type: ignore[no-untyped-def]
    return dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-11, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )


def test_idle_and_zero_envelope_segments_are_the_exact_propagator(raman) -> None:  # type: ignore[no-untyped-def]
    """A 300 us idle and a zero-envelope Stark probe between two pi/2 pulses integrate as ``exact``, hold P1 to 1e-12 and give
    e^{-iH tau} of the builder's constant Hamiltonian to 1e-10."""
    dev, dd, space = raman
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    t_half = 0.5 * math.pi / om
    drive = square_drive(dd, detuning_hz=1e3, include_stark=False)
    probe = Drive(
        kind="raman",
        ions=(0,),
        tones=(Tone(detuning_hz=0.0, phase_rad=0.0, envelope_hz=0.0),),
        beams=drive.beams,
        stark_shift_hz=-120.0,
        crosstalk={},
    )
    t1, t2, t3 = t_half, t_half + 300e-6, t_half + 300e-6 + 250e-6
    p1, p_stark = Pulse(drive, 0.0, t1, "p1", ()), Pulse(probe, t2, t3, "stark", ())
    sched = Schedule((p1, p_stark, Pulse(drive, t3, t3 + t_half, "p2", ())), ((t1, t2),), (), {0: 0.0})
    state = space.initial_state([0], fock={KX: 1})
    full, eng = _run(dev, sched, state, space)
    assert [s.integrator for s in eng.last_report.segments] == ["dop853", "exact", "exact", "dop853"]
    assert np.ptp(np.real(full.expectations["P1[0]"])[3:10]) < 1e-12, "no population moves in an idle"
    after_p1, _ = _run(dev, Schedule((p1,), (), (), {0: 0.0}), state, space)
    through_probe, _ = _run(dev, Schedule((p1, p_stark), ((t1, t2),), (), {0: 0.0}), state, space)
    h_idle = build_hamiltonian(dev, [], space, sample=quiet_sample()).H(t1)
    h_probe = build_hamiltonian(dev, [p_stark], space, sample=quiet_sample()).H(t2)
    assert h_idle.isherm and h_probe.isherm
    expected = (-1j * (t3 - t2) * h_probe).expm() * (-1j * (t2 - t1) * h_idle).expm() * after_p1.final.joint
    assert (through_probe.final.joint - expected).norm() < 1e-10


def test_a_ramsey_scan_with_millisecond_delays_costs_milliseconds() -> None:
    """A three-point Ramsey scan with delays up to 1 ms runs in under 5 s and shows the 1 kHz fringe."""
    fx = yb171_chain(2)
    t0 = time.perf_counter()
    res = ramsey(Machine(fx.device), 0, [0.0, 0.5e-3, 1e-3], nbar={2: 0.0185, 3: 0.0154}, detuning_hz=1e3)
    assert time.perf_counter() - t0 < 5.0
    p1 = res.data[:, 1]
    assert p1[0] > 0.99 and p1[1] < 0.02 and p1[2] > 0.99, (
        "a 1 kHz fringe: maxima at 0 and 1 ms, a minimum at 0.5 ms"
    )


def test_heating_idle_in_the_rotating_frame_matches_the_master_equation(raman) -> None:  # type: ignore[no-untyped-def]
    """A heated idle integrates in the frame rotating with H_mot and matches the Schroedinger-picture master equation (state to
    1e-6, P1 and n to 1e-7) while the mode heats."""
    dev, dd, space = raman
    noisy = _heated(dev)
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    t_half = 0.5 * math.pi / om
    idle = 20e-6
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, t_half, "p1", ())
    state = space.initial_state([0], thermal={KX: 0.3})
    opts = Numerics(lindblad_method="mesolve", mesolve_dimension_max=10**6)
    kw = {"device_channels": True}
    after_pulse, eng_p = _run(noisy, Schedule((pulse,), (), (), {0: 0.0}), state, space, opts, **kw)
    idled, eng = _run(
        noisy, Schedule((pulse,), ((t_half, t_half + idle),), (), {0: 0.0}), state, space, opts, **kw
    )
    assert eng.last_report.segments[1].integrator.startswith("dop853[rotating frame]")
    # the margin monitor raised the cap during the pulse; the idle runs on the grown space
    sp = eng.last_report.space
    assert eng_p.last_report.space == sp
    times = np.linspace(t_half, t_half + idle, 4)
    ref = evolve(
        build_hamiltonian(noisy, [], sp, sample=quiet_sample()).H,
        after_pulse.final.joint,
        times,
        c_ops=[c.op for c in noisy.noise.channels(noisy, sp)],
        e_ops={"P1[0]": sp.projector(0, 1), f"n[{KX}]": sp.number(KX)},
        options=opts,
    )
    assert (idled.final.joint - ref.final).norm() < 1e-6
    for key in ("P1[0]", f"n[{KX}]"):
        assert np.max(np.abs(idled.expectations[key][-4:] - ref.expect[key])) < 1e-7
    assert idled.final.motional.nbar[KX] > after_pulse.final.motional.nbar[KX], (
        "the mode heats during the idle"
    )


def test_a_mixture_without_dissipation_is_evolved_as_weighted_pure_branches(raman) -> None:  # type: ignore[no-untyped-def]
    """A thermal mode under a carrier pulse evolves as sesolve eigen-branches that reproduce the master equation (P1 and state to
    1e-6, n to 1e-5) and drop weight below the threshold with a note; a heated mixture keeps mesolve."""
    dev, dd, space = raman
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    state = space.initial_state([0], thermal={KX: 0.8})
    kets, weights, dropped = _pure_branches(state.joint, 1e-6)
    assert dropped == 0.0 and abs(sum(weights) - 1.0) < 1e-12 and weights == sorted(weights, reverse=True)
    rebuilt = sum((w * k.proj() for k, w in zip(kets, weights)), 0.0 * kets[0].proj())
    assert (rebuilt - state.joint).norm() < 1e-12
    kets3, weights3, dropped3 = _pure_branches(state.joint, 0.05)
    assert len(kets3) < len(kets) and dropped3 > 0.0 and abs(sum(weights3) - 1.0) < 1e-12
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, 2.5 * math.pi / om, "p", ())
    sched = Schedule((pulse,), (), (), {0: 0.0})
    fast, eng = _run(dev, sched, state, space)
    assert eng.last_report.method == "sesolve" and eng.last_report.trajectories == len(kets)
    # the boundary monitor raised the cap: the reference integrates the regridded mixture on the grown space
    sp = eng.last_report.space
    ref = evolve(
        build_hamiltonian(dev, [pulse], sp, sample=quiet_sample()).H,
        regrid_state(state.joint, space, sp) if sp != space else state.joint,
        fast.times_s,
        e_ops={"P1[0]": sp.projector(0, 1)},
        options=Numerics(),
    )
    assert np.max(np.abs(np.real(fast.expectations["P1[0]"]) - np.real(ref.expect["P1[0]"]))) < 1e-6
    assert (fast.final.internal - sp.internal_marginal(ref.final)).norm() < 1e-6
    assert abs(fast.final.motional.nbar[KX] - float(np.real(qt.expect(sp.number(KX), ref.final)))) < 1e-5
    # the threshold drops branches and says so
    _, eng = _run(dev, sched, state, space, Numerics(branch_weight_min=0.05))
    assert eng.last_report.trajectories == len(kets3)
    assert any("pure branches" in n and "dropped" in n for n in eng.last_report.notes)
    # a density matrix WITH collapse operators keeps the master equation
    _, eng_c = _run(
        _heated(dev),
        sched,
        state,
        space,
        Numerics(lindblad_method="mesolve", mesolve_dimension_max=10**6),
        device_channels=True,
    )
    assert eng_c.last_report.method == "mesolve" and eng_c.last_report.trajectories == 1


def test_drive_coefficients_are_the_plans_tone_sum(raman) -> None:  # type: ignore[no-untyped-def]
    """The plain and conjugate drive terms carry (1/2) Omega e^{-i(mu t - phi)} times the static factors to 1e-12, and every
    interaction-picture sideband term has the same modulus."""
    dev, dd, space = raman
    drive = square_drive(dd, detuning_hz=12.5e3, phase_rad=0.3, include_stark=False)
    pulse = Pulse(drive, 2e-6, 12e-6, "p", ())
    for bopts in (BuilderOptions(), BuilderOptions(frame="interaction")):
        built = build_hamiltonian(dev, [pulse], space, sample=quiet_sample(), options=bopts)
        pairs = [item for item in built.H.to_list() if isinstance(item, list)]
        rec = built.records[0]
        for t in (2.5e-6, 7.1e-6, 11.9e-6):
            tone = drive.tones[0]
            ref = (
                0.5
                * 2.0
                * math.pi
                * float(tone.envelope_hz)
                * np.exp(-1j * (2.0 * math.pi * float(tone.detuning_hz) * t - float(tone.phase_rad)))
            )  # type: ignore[arg-type]
            ref *= rec.debye_waller * rec.carrier_factor
            if bopts.frame == "schrodinger":
                assert len(pairs) == 2
                assert abs(pairs[0][1](t) - ref) < 1e-12 * abs(ref) and abs(
                    pairs[1][1](t) - np.conj(ref)
                ) < 1e-12 * abs(ref)
            else:
                # one plain and one conjugate element per sideband operator QobjEvo keeps (it drops the far corners of D,
                # whose elements tidy to zero); every term carries the tone sum times a phase
                pic = interaction_picture(space, 0, {KX: rec.etas[KX]})
                n_significant = sum(1 for term in pic.terms if term.op.norm() > 1e-12)
                assert 2 * n_significant <= len(pairs) <= 2 * len(pic.terms) and n_significant >= 11
                assert all(abs(abs(p[1](t)) - abs(ref)) < 1e-12 * abs(ref) for p in pairs), (
                    "every sideband term is the tone sum times a phase"
                )


def test_simultaneous_pulses_of_unequal_length_are_integrated_segment_by_segment() -> None:
    """Two pi/2 pulses that start together and end 1% apart run as two segments, both ions reach P1 = 0.5 to 1e-6 and the
    shorter pulse's ion holds (1e-9) after it ends."""
    dev = two_ion_raman_device()
    space = HilbertSpace((2, 2), (), None, tuple(range(len(dev.crystal.modes))))
    rabi = {0: 100e3, 1: 101e3}
    pulses = tuple(
        Pulse(square_microwave_drive(q, rabi[q]), 0.0, 0.25 / rabi[q], f"pi_half/ion{q}", ()) for q in (0, 1)
    )
    sched = Schedule(pulses, (), (), {0: 0.0, 1: 0.0})
    eng = JointExactEngine(store_per_segment=2)
    tr = eng.run_pulses(
        dev, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), Numerics()
    )
    assert len(eng.last_report.segments) == 2
    for q in (0, 1):
        assert np.real(tr.expectations[f"P1[{q}]"][-1]) == pytest.approx(0.5, abs=1e-6)
    # the shorter pulse has ended when the longer one's second segment runs: its ion holds its population
    t_short = 0.25 / rabi[1]
    idx = np.flatnonzero(tr.times_s >= t_short - 1e-15)
    assert np.ptp(np.real(tr.expectations["P1[1]"][idx])) < 1e-9


# ---- tolerance convergence and the integrator ladder (Sections 5.3, 5.5) -----------------------------------------------------


def _runner(dev, drive, t_end, space, n0=0):  # type: ignore[no-untyped-def]
    """A closure that integrates one pulse under the Numerics it is handed and returns its population traces."""

    def run(options: Numerics) -> dict[str, np.ndarray]:
        eng = JointExactEngine(store_per_segment=9)
        tr = eng.run_pulses(
            dev,
            Schedule((Pulse(drive, 0.0, t_end, "p", ()),), (), (), {0: 0.0}),
            space.initial_state([0], fock={KX: n0}),
            space,
            quiet_sample(),
            SeedSpec(0),
            options,
        )
        return {k: np.asarray(v) for k, v in tr.expectations.items() if k.startswith("P")}

    return run


def test_tolerance_convergence_on_the_carrier_and_sideband_fixtures(raman) -> None:  # type: ignore[no-untyped-def]
    """Carrier and blue-sideband pi pulses change by less than 1e-6 when the tolerances (1e-10, 1e-8) are tightened ten-fold."""
    dev, dd, _space = raman
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 3), 0.2),), None, (0, 2))
    om = TWO_PI * dd.carrier_rabi_hz
    eta = dd.etas[KX]
    rep = convergence_check(_runner(dev, square_drive(dd, include_stark=False), math.pi / om, space))
    assert rep.tolerances == (1e-10, 1e-8)
    assert rep.tightened_tolerances == pytest.approx((1e-11, 1e-9), rel=1e-12)
    assert rep.converged and rep.max_change < 1e-6, rep.summary()
    assert "tolerance convergence" in rep.summary() and "converged" in rep.summary()
    t_blue = math.pi / sideband_rabi_rad_s(om, eta, 0, 1)
    rep_b = convergence_check(
        _runner(dev, square_drive(dd, detuning_hz=3.0e6, include_stark=False), t_blue, space)
    )
    assert rep_b.converged and rep_b.max_change < 1e-6, rep_b.summary()


def test_convergence_check_rejects_a_mismatched_observable_set_and_a_bad_factor(raman) -> None:  # type: ignore[no-untyped-def]
    dev, dd, space = raman
    om = TWO_PI * dd.carrier_rabi_hz
    run = _runner(dev, square_drive(dd, include_stark=False), math.pi / om, space)
    with pytest.raises(ValueError, match="tightening factor"):
        convergence_check(run, Numerics(), factor=1.0)

    def wobbly(options: Numerics) -> dict[str, np.ndarray]:
        out = run(options)
        if options.atol < 1e-10:
            out["extra"] = np.zeros(1)
        return out

    with pytest.raises(ValueError, match="different observables"):
        convergence_check(wobbly, Numerics())


def test_a_deliberate_tolerance_survives_the_large_mode_atol_keying() -> None:
    """At d_m = 121 the default atol relaxes to LARGE_MODE_ATOL (at d_m = 100 it stays 1e-10) while a tightened (1e-11) or
    loosened (1e-6) one is kept."""
    d = 121
    assert d > LARGE_MODE_DIMENSION
    h = WX * qt.num(d)
    psi0 = qt.basis(d, 1)
    times = np.linspace(0.0, 1e-6, 5)
    default = evolve(h, psi0, times, options=Numerics(), largest_mode_dimension=d)
    assert default.atol == LARGE_MODE_ATOL
    at_threshold = evolve(h, psi0, times, options=Numerics(), largest_mode_dimension=LARGE_MODE_DIMENSION)
    assert at_threshold.atol == 1e-10
    tight = tightened(Numerics())
    got = evolve(h, psi0, times, options=tight, largest_mode_dimension=d)
    assert got.atol == tight.atol == pytest.approx(1e-11, rel=1e-12)
    loose = evolve(h, psi0, times, options=Numerics(atol=1e-6), largest_mode_dimension=d)
    assert loose.atol == 1e-6


@pytest.mark.slow
def test_the_ladder_at_the_large_caps() -> None:
    """A spin-dependent force at d_m = 101, 121, 151 and 201 integrates on dop853 with no retry, atol LARGE_MODE_ATOL and norm 1
    to 1e-7, and at d_m = 100 atol stays 1e-10."""
    eta, om_drive = 0.1, TWO_PI * 250e3
    delta = WX - TWO_PI * 20e3
    for d in (101, 121, 151, 201):
        disp = qt.displace(d, 1j * eta)
        a = qt.destroy(d)
        h = qt.QobjEvo(
            [
                qt.tensor(qt.qeye(2), WX * a.dag() * a),
                [
                    qt.tensor(qt.sigmap(), disp),
                    lambda t, **kw: 0.5 * om_drive * np.exp(-1j * delta * t),
                ],
                [
                    qt.tensor(qt.sigmap(), disp).dag(),
                    lambda t, **kw: 0.5 * om_drive * np.exp(1j * delta * t),
                ],
            ]
        )
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(d, 0))
        ev = evolve(h, psi0, [0.0, 10e-6], options=Numerics(), largest_mode_dimension=d, omega_max_rad_s=WX)
        assert ev.integrator == "dop853" and ev.retries == (), (d, ev.retries)
        assert ev.atol == LARGE_MODE_ATOL, d
        assert ev.final.norm() == pytest.approx(1.0, abs=1e-7), d
    d = LARGE_MODE_DIMENSION
    at_threshold = evolve(
        WX * qt.num(d),
        qt.basis(d, 1),
        np.linspace(0.0, 1e-6, 3),
        options=Numerics(),
        largest_mode_dimension=d,
    )
    assert at_threshold.atol == 1e-10


# ---- the per-time Fock marginals and the wall times of Traces -----------------------------------------------------------------


@pytest.fixture(scope="module")
def stored() -> tuple[Machine, trap.Result]:
    fx = yb171_chain(2)
    machine = Machine(
        fx.device,
        numerics=trap.Numerics(branch_weight_min=1e-3, store_marginals=True),
    ).calibrated(pairs=[(0, 1)], detection_records=500)
    return machine, machine.run(BELL, 100, seed=0)


def test_the_marginal_is_a_distribution_whose_mean_is_the_occupation_at_every_stored_time(
    stored: tuple[Machine, trap.Result],
) -> None:
    _machine, result = stored
    rec = last_record(result)
    assert rec.traces, "a JOINT_EXACT run stores one trace per (sample, branch)"
    for tr in rec.traces:
        assert tr.mode_marginal is not None and set(tr.mode_marginal) == set(tr.mode_occupations)
        for m, dist in tr.mode_marginal.items():
            d_m = next(t.d for t in result.diagnostics.space.resolved if t.mode == m)
            assert dist.shape == (tr.times_s.size, d_m), (m, dist.shape)
            assert np.all(dist >= -1e-12)
            # the integrator runs with normalize_output off: the norm drifts by the solver tolerance
            assert np.max(np.abs(dist.sum(axis=1) - 1.0)) < 1e-6
            mean = dist @ np.arange(d_m)
            assert np.max(np.abs(mean - tr.mode_occupations[m])) < 1e-8, m
            # the last row is the final reduced motional state's diagonal
            final = np.real(np.diag(np.asarray(tr.final.motional.reduced[m].full())))
            assert np.max(np.abs(dist[-1] - final)) < 1e-9


def test_the_marginal_is_off_by_default_and_changes_no_number(stored: tuple[Machine, trap.Result]) -> None:
    machine, result = stored
    plain = Machine(machine.device, table=machine.table, numerics=trap.Numerics(branch_weight_min=1e-3))
    assert not plain.numerics.store_marginals and not plain.numerics.store_marginals
    assert machine.numerics.store_marginals
    quiet = plain.run(BELL, 100, seed=0)
    for tr in last_record(quiet).traces:
        assert tr.mode_marginal is None
    assert np.array_equal(quiet.bitstrings, result.bitstrings)


def test_wall_time_is_keyed_by_the_pulses_and_the_segments_carry_it(
    stored: tuple[Machine, trap.Result],
) -> None:
    machine, result = stored
    rec = last_record(result)
    gate_ids = {p.gate_id for p in rec.schedule.pulses}
    for tr in rec.traces:
        assert tr.wall_time_s and set(tr.wall_time_s) <= gate_ids | {"idle"}
        assert gate_ids <= set(tr.wall_time_s), "every pulse of the schedule was integrated and timed"
        assert all(v >= 0.0 for v in tr.wall_time_s.values()) and sum(tr.wall_time_s.values()) > 0.0
    # the same pulses through the engine directly, from one pure branch: the per-segment numbers sum to the per-pulse ones
    space = result.diagnostics.space
    opts = machine.numerics
    state = space.initial_state([0] * machine.device.crystal.n_ions, fock={t.mode: 0 for t in space.resolved})
    engine = machine.engine
    traces = engine.run_pulses(machine.device, rec.schedule, state, space, quiet_sample(0), SeedSpec(0), opts)
    report = engine.last_report
    assert report is not None and report.segments
    assert all(seg.wall_time_s > 0.0 for seg in report.segments)
    assert sum(seg.wall_time_s for seg in report.segments) == pytest.approx(
        sum(traces.wall_time_s.values()), rel=1e-9
    )
    assert traces.mode_marginal is not None and set(traces.mode_marginal) == set(traces.mode_occupations)
