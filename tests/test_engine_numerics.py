"""The engine's exact shortcuts against the ODE reference (PLAN.md Sections 5.2, 5.3, 5.5; the M8 speed work).

Three paths exist because an ODE solve of an oscillatory Hamiltonian at 14 to 45 steps per mode period is the wrong tool where
the propagator is known in closed form or the state is a mixture without dissipation, and each is pinned here against the path
it replaces: (i) a constant Hamiltonian (an idle interval, a zero-envelope pulse carrying only its light shift) is propagated by
its exact phases, and with the device's heating operators present by the master equation in the frame rotating with H_mot,
where the dissipators are invariant; (ii) a density matrix evolved without collapse operators is the weighted sum of its pure
eigen-branches through ``sesolve`` (the Fock-sum path), never the D^2 Liouvillian; (iii) the plain and conjugate drive terms and
every sideband term of the interaction picture share one tone-sum evaluation per time. None of them is an approximation beyond
the declared branch threshold, so the results agree with the reference to the solver tolerance.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.api import white_spectrum
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions, _pure_branches
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.experiments import ramsey
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import quiet_sample
from tests.m2_fixtures import single_ion_raman_device

KX = 1


@pytest.fixture(scope="module")
def raman():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 6), 0.2),), None, (0, 2))
    return dev, dd, space


def _both(dev, sched, state, space, **engine_kw):  # type: ignore[no-untyped-def]
    """The same schedule through the shortcut and through the ODE reference: (shortcut traces, reference traces, engines)."""
    out = []
    engines = []
    for flags in ({}, {"closed_form_constant": False, "pure_branches": False}):
        eng = JointExactEngine(store_per_segment=4, **{**engine_kw, **flags})
        out.append(eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions()))
        engines.append(eng)
    return out[0], out[1], engines


def test_idle_and_zero_envelope_segments_use_the_exact_propagator_and_match_the_ode(raman) -> None:  # type: ignore[no-untyped-def]
    """A pi/2 pulse, a 300 us idle and a zero-envelope Stark probe: the same populations as the ODE ladder to the solver
    tolerance, in a few milliseconds instead of hundreds of thousands of right-hand sides; the idle reports ``exact``."""
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
    sched = Schedule(
        (
            Pulse(drive, 0.0, t1, "p1", ()),
            Pulse(probe, t2, t3, "stark", ()),
            Pulse(drive, t3, t3 + t_half, "p2", ()),
        ),
        ((t1, t2),),
        (),
        {0: 0.0},
    )
    state = space.initial_state([0], fock={KX: 1})
    fast, ref, (eng_fast, eng_ref) = _both(dev, sched, state, space)
    # the ODE drifts by ~2e-7 over 300 us of free rotation (its tolerance); the closed form holds P1 exactly constant there
    assert np.max(np.abs(np.real(fast.expectations["P1[0]"]) - np.real(ref.expectations["P1[0]"]))) < 1e-6
    assert np.ptp(np.real(fast.expectations["P1[0]"])[3:10]) < 1e-12, "no population moves in an idle"
    assert (fast.final.joint - ref.final.joint).norm() < 5e-6
    labels = [s.integrator for s in eng_fast.last_report.segments]
    assert labels == ["dop853", "exact", "exact", "dop853"], labels
    assert all(s.integrator == "dop853" for s in eng_ref.last_report.segments)
    assert eng_fast.last_report.segments[1].rhs_evaluations is None


def test_a_ramsey_scan_with_millisecond_delays_costs_milliseconds() -> None:
    """The Ramsey experiments of Section 7.5 (field, Stark, qubit frequency) idle for up to 2 ms; the closed form makes them cheap."""
    import time

    from tests.m6_fixtures import circuit_fixture

    fx = circuit_fixture(2)
    t0 = time.perf_counter()
    res = ramsey(
        fx.device,
        0,
        [0.0, 0.5e-3, 1e-3],
        gate_drive=fx.gate_drives[0],
        nbar={2: 0.0185, 3: 0.0154},
        detuning_hz=1e3,
    )
    assert time.perf_counter() - t0 < 5.0
    p1 = res.data[:, 1]
    assert p1[0] > 0.99 and p1[1] < 0.02 and p1[2] > 0.99, (
        "a 1 kHz fringe: maxima at 0 and 1 ms, a minimum at 0.5 ms"
    )


def test_heating_idle_in_the_rotating_frame_matches_the_ode_reference(raman) -> None:  # type: ignore[no-untyped-def]
    """With the device's heating operators (eigenoperators of ad_H) an idle is integrated in the frame rotating with H_mot; the
    reduced states, occupations and coherences agree with the Schroedinger-picture master equation."""
    dev, dd, space = raman
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-11, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    t_half = 0.5 * math.pi / om
    drive = square_drive(dd, include_stark=False)
    idle = 20e-6
    sched = Schedule(
        (Pulse(drive, 0.0, t_half, "p1", ()), Pulse(drive, t_half + idle, 2 * t_half + idle, "p2", ())),
        ((t_half, t_half + idle),),
        (),
        {0: 0.0},
    )
    state = space.initial_state([0], thermal={KX: 0.3})
    opts = SolverOptions(lindblad_method="mesolve", mesolve_dimension_max=10**6)
    traces = {}
    labels = {}
    for flag in (True, False):
        eng = JointExactEngine(device_channels=True, closed_form_constant=flag, store_per_segment=3)
        traces[flag] = eng.run_pulses(noisy, sched, state, space, quiet_sample(), SeedSpec(0), opts)
        labels[flag] = [s.integrator for s in eng.last_report.segments]
    assert labels[True][1].startswith("dop853[rotating frame]") and labels[False][1] == "dop853"
    assert (traces[True].final.joint - traces[False].final.joint).norm() < 1e-6
    for key in ("P1[0]", f"n[{KX}]"):
        assert np.max(np.abs(traces[True].expectations[key] - traces[False].expectations[key])) < 1e-7
    assert traces[True].final.motional.nbar[KX] > 0.3, "the mode heats during the idle"


def test_a_mixture_without_dissipation_is_evolved_as_weighted_pure_branches(raman) -> None:  # type: ignore[no-untyped-def]
    """A thermal mode under a carrier pulse: the eigen-branches through sesolve reproduce the density-matrix reference, the
    weights renormalize and the dropped weight is reported (Section 5.3, the Fock-sum path)."""
    dev, dd, space = raman
    om = 2.0 * math.pi * dd.carrier_rabi_hz
    state = space.initial_state([0], thermal={KX: 0.8})
    kets, weights, dropped = _pure_branches(state.joint, 1e-6)
    assert dropped == 0.0 and abs(sum(weights) - 1.0) < 1e-12 and weights == sorted(weights, reverse=True)
    rebuilt = sum((w * k.proj() for k, w in zip(kets, weights)), 0.0 * kets[0].proj())
    assert (rebuilt - state.joint).norm() < 1e-12
    kets3, weights3, dropped3 = _pure_branches(state.joint, 0.05)
    assert len(kets3) < len(kets) and dropped3 > 0.0 and abs(sum(weights3) - 1.0) < 1e-12
    sched = Schedule(
        (Pulse(square_drive(dd, include_stark=False), 0.0, 2.5 * math.pi / om, "p", ()),), (), (), {0: 0.0}
    )
    fast, ref, (eng_fast, eng_ref) = _both(dev, sched, state, space)
    assert eng_fast.last_report.method == "sesolve" and eng_fast.last_report.trajectories == len(kets)
    assert eng_ref.last_report.method == "mesolve" and eng_ref.last_report.trajectories == 1
    assert np.max(np.abs(np.real(fast.expectations["P1[0]"]) - np.real(ref.expectations["P1[0]"]))) < 1e-6
    assert (fast.final.internal - ref.final.internal).norm() < 1e-6
    assert abs(fast.final.motional.nbar[KX] - ref.final.motional.nbar[KX]) < 1e-5
    # the threshold drops branches and says so
    eng = JointExactEngine()
    eng.run_pulses(
        dev, sched, state, space, quiet_sample(), SeedSpec(0), SolverOptions(branch_weight_min=0.05)
    )
    assert eng.last_report.trajectories == len(kets3)
    assert any("pure branches" in n and "dropped" in n for n in eng.last_report.notes)
    # a density matrix WITH collapse operators keeps the reference path
    eng_c = JointExactEngine(device_channels=True)
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-11, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    eng_c.run_pulses(
        noisy,
        sched,
        state,
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(lindblad_method="mesolve", mesolve_dimension_max=10**6),
    )
    assert eng_c.last_report.method == "mesolve" and eng_c.last_report.trajectories == 1


def test_drive_coefficients_are_shared_and_exact(raman) -> None:  # type: ignore[no-untyped-def]
    """The plain and conjugate terms (and the sideband terms of the interaction picture) evaluate one tone sum per time; the
    values are the plan's formula (1/2) sum_tones Omega e^{-i(mu t - phi)} times the static factors, to round-off."""
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
                from qutip_trap.dynamics.frames import interaction_picture

                # one plain and one conjugate element per sideband operator QobjEvo keeps (it drops the far corners of D,
                # whose elements tidy to zero); every term carries the tone sum times a phase
                pic = interaction_picture(space, 0, {KX: rec.etas[KX]})
                n_significant = sum(1 for term in pic.terms if term.op.norm() > 1e-12)
                assert 2 * n_significant <= len(pairs) <= 2 * len(pic.terms) and n_significant >= 11
                assert all(abs(abs(p[1](t)) - abs(ref)) < 1e-12 * abs(ref) for p in pairs), (
                    "every sideband term is the tone sum times a phase"
                )
        calls0 = built.counter.calls
        built.H(3e-6)  # one evaluation of every element at one time
        assert built.counter.calls - calls0 == len(pairs), (
            "one counter increment per coefficient-bearing element"
        )


def test_simultaneous_pulses_of_unequal_length_are_integrated_segment_by_segment() -> None:
    """Two ions' pi/2 pulses at two FITTED Rabi frequencies start together and end 1 % apart (the parity scan's analysis
    pulses on a calibrated table): the engine cuts the schedule at both ends and the builder takes the pulses that span each
    segment, each on its own clock; every ion flops as sin^2(Omega_i t/2) and the first M8 build refused the second segment."""
    from qutip_trap.light.microwave import square_microwave_drive
    from tests.m2_fixtures import two_ion_raman_device

    dev = two_ion_raman_device()
    space = HilbertSpace((2, 2), (), None, tuple(range(len(dev.crystal.modes))))
    rabi = {0: 100e3, 1: 101e3}
    pulses = tuple(
        Pulse(square_microwave_drive(q, rabi[q]), 0.0, 0.25 / rabi[q], f"pi_half/ion{q}", ()) for q in (0, 1)
    )
    sched = Schedule(pulses, (), (), {0: 0.0, 1: 0.0})
    eng = JointExactEngine(store_per_segment=2)
    tr = eng.run_pulses(
        dev, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    assert len(eng.last_report.segments) == 2
    for q in (0, 1):
        assert np.real(tr.expectations[f"P1[{q}]"][-1]) == pytest.approx(0.5, abs=1e-6)
    # the shorter pulse has ended when the longer one's second segment runs: ion 0 holds its population
    t_short = 0.25 / rabi[1]
    idx = np.flatnonzero(tr.times_s >= t_short - 1e-15)
    assert np.ptp(np.real(tr.expectations["P1[1]"][idx])) < 1e-9
