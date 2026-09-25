"""ms and zz through the scheduler (PLAN.md Section 7): the native matrices, sign and axis conventions, the ZZ wrapper,
virtual-Z frames, partial angles, refusals, the IonQ JSON path, and one entangling gate at a time per crystal."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration.entangling import calibrate_entangling_angle, frame_rotated, gate_space
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.native import equal_up_to_global_phase, gpi2, rz, xx
from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.native import zz as native_zz
from qutip_trap.control.pulses import LightShiftCouplings
from qutip_trap.control.schedule import GateDrive, PhaseFrame, ScheduleError, ms_spin_phases, schedule
from qutip_trap.control.shaping import gate_modes, solve_amplitude_modulation, symmetric_pulse
from qutip_trap.control.table import Waveform
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.io.ionq import load_ionq_json
from qutip_trap.noise.sampling import quiet_sample
from tests.fixtures import (
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    table_with_waveform,
    two_ion_modes,
)

RABI_TABLE, STARK_TABLE = derived_seeds(chain_device(2), raman_gate_drives(2))
"""The derived carrier Rabi frequencies and differential Stark shifts a calibrated table of this device holds (the played
chain is then the identity; the scheduler compensates the Stark shifts)."""
MS = math.pi / 2.0


@pytest.fixture(scope="module")
def calibrated():  # type: ignore[no-untyped-def]
    """The two-mode AM gate, exactly calibrated to chi = pi/4, with its table, space and drives."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    am = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    space = gate_space(modes, 2, waveform=am.waveform)
    table0 = table_with_waveform((0, 1), am.waveform, rabi_hz=RABI_TABLE, stark_hz=STARK_TABLE)
    run = calibrate_entangling_angle(
        dev, am.waveform, (0, 1), drives, table0, space=space, tolerance_rad=2e-4
    )
    assert run.converged
    table = table_with_waveform((0, 1), run.waveform, rabi_hz=RABI_TABLE, stark_hz=STARK_TABLE)
    return dev, modes, drives, space, table, run


def _run_circuit(dev, table, space, ops, internal=(0, 0), **kw):  # type: ignore[no-untyped-def]
    circ = Circuit(2, tuple(ops), (0, 1))
    sch = schedule(circ, dev, table, **kw)
    eng = JointExactEngine(table=table)
    tr = eng.run_pulses(
        dev,
        sch,
        space.initial_state(list(internal) if not isinstance(internal, qt.Qobj) else internal),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    return tr.final.internal, sch


def _fidelity(
    rho: qt.Qobj, mat: np.ndarray, ket0: np.ndarray, frame: dict[int, float] | None = None
) -> float:
    """Overlap with mat |ket0>, the target rotated by the schedule's final virtual-Z frame when given (the compensated light shifts'
    RZ(2 pi int delta dt) the state carries and the frame absorbs)."""
    target = qt.Qobj(mat @ ket0)
    target.dims = [[2, 2], [1, 1]]
    if frame:
        target = frame_rotated(target, frame)
    return float(np.real(qt.expect(rho, target)))


KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def test_ms_reproduces_the_native_matrix_for_arbitrary_phases(calibrated) -> None:  # type: ignore[no-untyped-def]
    """MS(phi_0, phi_1, pi/2) on |00> reproduces the native matrix to 1.1 times the spot check's infidelity at three phase pairs
    and overlaps the opposite sign by less than 0.02; a positive kernel plays pi on the second ion."""
    dev, _modes, _drives, space, table, run = calibrated
    budget = 1.0 - run.checks[-1].fidelity
    assert budget < 1e-3
    for phi0, phi1 in ((0.0, 0.0), (0.3, 1.1), (-0.7, 2.0)):
        rho, sch = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (phi0, phi1, math.pi / 2))])
        # the played gate reproduces the native matrix to the exact spot check's own infidelity (measured 2.2731e-04
        # against the spot check's 2.2725e-04)
        infidelity = 1.0 - _fidelity(rho, native_ms(phi0, phi1, math.pi / 2), KET00, sch.phase_frame)
        assert infidelity < 1.1 * budget, (infidelity, budget)
        assert _fidelity(rho, native_ms(phi0, phi1, -math.pi / 2), KET00, sch.phase_frame) < 0.02
        assert len(sch.pulses) == 10 and all(p.closes_modes == (2, 3) for p in sch.pulses)
    # a positive kernel sign is played with pi on the second ion (exp(+i chi sigma sigma) = XX(-chi))
    wf = table.waveform_for((0, 1))
    assert wf is not None and wf.chi_total_rad > 0.0
    spins, chi_abs = ms_spin_phases(wf, (0, 1), (0.0, 0.0), PhaseFrame())
    assert spins[0] == pytest.approx(-math.pi / 2) and spins[1] == pytest.approx(math.pi / 2)
    assert chi_abs == pytest.approx(wf.chi_total_rad)


def test_partial_angle_rescales_by_the_s_squared_law(calibrated) -> None:  # type: ignore[no-untyped-def]
    """MS(0, 0, +-0.6) scales every amplitude by sqrt((theta/2)/chi) to 1e-12, gives P_11 = sin^2(theta/2) to 3e-3 and
    matches the native matrix above 0.995."""
    dev, _modes, _drives, space, table, _run = calibrated
    theta = 0.6
    rho, sch = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (0.0, 0.0, theta))])
    wf = table.waveform_for((0, 1))
    assert wf is not None
    played = sch.pulses[0].drive.tones[0].envelope_hz
    assert abs(wf.chi_total_rad - math.pi / 4) < 2e-4, "the calibrated waveform carries its exact angle"
    amp0 = wf.segments[0].amplitude_hz[(0, "blue")]
    assert not callable(amp0) and not callable(played)
    expected = float(amp0) * math.sqrt((theta / 2) / abs(wf.chi_total_rad))
    assert float(played) == pytest.approx(expected, rel=1e-12)
    p11 = float(np.real(rho.full()[3, 3]))
    assert p11 == pytest.approx(math.sin(theta / 2) ** 2, abs=3e-3)
    assert _fidelity(rho, native_ms(0.0, 0.0, theta), KET00, sch.phase_frame) > 0.995
    # a negative angle is a pi on the second phase
    rho_n, sch_n = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (0.0, 0.0, -theta))])
    assert _fidelity(rho_n, native_ms(0.0, 0.0, -theta), KET00, sch_n.phase_frame) > 0.995


def test_zz_wrapper_construction_matrix_and_schedule(calibrated) -> None:  # type: ignore[no-untyped-def]
    """ZZ(theta) = [GPi2(pi/2) (x) GPi2(pi/2)] XX(theta/2) [GPi2(3 pi/2) (x) GPi2(3 pi/2)] exactly, and the scheduled wrappers,
    ten MS segments and dead times reproduce ZZ(pi/2) to 2.6 times the spot check's infidelity."""
    w = np.kron(gpi2(math.pi / 2), gpi2(math.pi / 2))
    w_in = np.kron(gpi2(1.5 * math.pi), gpi2(1.5 * math.pi))
    for theta in (math.pi / 2, 0.8, -0.4):
        assert equal_up_to_global_phase(w @ xx(theta / 2) @ w_in, native_zz(theta))
    assert np.allclose(
        gpi2(math.pi / 2) @ np.array([[0, 1], [1, 0]]) @ gpi2(math.pi / 2).conj().T,
        -np.array([[1, 0], [0, -1]]),
    )
    dev, _modes, _drives, space, table, run = calibrated
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    internal = qt.tensor(plus, plus)
    rho, sch = _run_circuit(dev, table, space, [Operation("zz", (0, 1), (math.pi / 2,))], internal=internal)
    ids = [p.gate_id for p in sch.pulses]
    assert (
        sum("wrap_in" in i for i in ids) == 2
        and sum("wrap_out" in i for i in ids) == 2
        and sum("/ms/" in i for i in ids) == 10
    )
    ket = np.asarray(internal.full()).ravel()
    budget = 1.0 - run.checks[-1].fidelity
    # the wrapper construction costs a factor 2.46 over the bare MS (measured 5.599e-04 against the spot check's
    # 2.2725e-04: four GPi2 pulses and three dead times around the entangling block)
    infidelity = 1.0 - _fidelity(rho, native_zz(math.pi / 2), ket, sch.phase_frame)
    assert infidelity < 2.6 * budget, (infidelity, budget)
    assert _fidelity(rho, native_zz(-math.pi / 2), ket, sch.phase_frame) < 0.05
    # the wrappers are 2.5 us GPi2 pulses at the table's 100 kHz with the dead time before and after the MS block
    wrap = [p for p in sch.pulses if "wrap_in" in (p.gate_id or "")]
    assert all(p.duration_s == pytest.approx(0.25 / RABI_TABLE[(0, 0)]) for p in wrap)
    ms_start = min(p.t_start_s for p in sch.pulses if "/ms/" in (p.gate_id or ""))
    assert ms_start == pytest.approx(max(p.t_end_s for p in wrap) + dev.hardware.dead_time_s)


def test_virtual_z_frame_carries_through_ms(calibrated) -> None:  # type: ignore[no-untyped-def]
    """RZ(0.4) on ion 1 then MS(0, 0) plays MS(0, -theta)|00> up to the frame (to three times the spot check's infidelity),
    while MS(0, 0) RZ|00> without the frame overlaps it by cos^2(theta/2)."""
    dev, _modes, _drives, space, table, run = calibrated
    theta = 0.4
    rho, sch = _run_circuit(
        dev, table, space, [Operation("rz", (1,), (theta,)), Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2))]
    )
    # the frame carries the virtual RZ(theta) on ion 1 plus, on both ions, the compensated light shift's 2 pi int delta dt
    stark_frame = {0: sch.phase_frame[0], 1: sch.phase_frame[1] - theta}
    assert sch.phase_frame[1] - sch.phase_frame[0] == pytest.approx(theta) and -0.1 < stark_frame[0] < 0.0
    assert stark_frame[1] == pytest.approx(stark_frame[0], abs=1e-9)
    budget = 1.0 - run.checks[-1].fidelity
    assert _fidelity(rho, native_ms(0.0, -theta, math.pi / 2), KET00, stark_frame) > 1.0 - 3 * budget
    circuit_order = native_ms(0.0, 0.0, math.pi / 2) @ np.kron(np.eye(2), rz(theta))
    assert (
        _fidelity(rho, np.kron(np.eye(2), rz(theta)).conj().T @ circuit_order, KET00, stark_frame)
        > 1.0 - 3 * budget
    )
    # a gate of infidelity eps moves a 0.96 overlap by up to about 2 sqrt(eps (1 - 0.96)): the negative control stays far from one
    assert _fidelity(rho, circuit_order, KET00, stark_frame) == pytest.approx(
        math.cos(theta / 2) ** 2, abs=4.0 * math.sqrt(budget)
    )
    # the full frame rotates the ideal circuit's state (virtual RZ included) onto the played one
    assert _fidelity(rho, circuit_order, KET00, sch.phase_frame) > 1.0 - 3 * budget


def test_scheduler_refusals_and_table_lookup(calibrated) -> None:  # type: ignore[no-untyped-def]
    dev, modes, drives, _space, table, _run = calibrated
    empty = table_with_waveform((0, 1), Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=20e3))

    none = dataclasses.replace(empty, ms={})
    with pytest.raises(ScheduleError, match="no entangling waveform"):
        schedule(Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1)), dev, none)
    # the reversed key is found
    assert dataclasses.replace(table, ms={(1, 0): table.ms[(0, 1)]}).waveform_for((0, 1)) is not None
    # a light-shift waveform cannot serve an MS gate, and needs a light-shift drive
    ls = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=20e3, kind="light_shift")
    ls_table = table_with_waveform((0, 1), ls, rabi_hz=RABI_TABLE, stark_hz=STARK_TABLE)
    with pytest.raises(ScheduleError, match="MS .* waveform"):
        schedule(Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1)), dev, ls_table)
    with pytest.raises(ScheduleError, match="light_shift gate drive"):
        schedule(Circuit(2, (Operation("zz", (0, 1), (-math.pi / 2,)),), (0, 1)), dev, ls_table)
    # a sigma_z force takes its sign from the detuning side only: the wrong sign is refused, not phase-flipped
    couplings = LightShiftCouplings((-2.0, 0.0), 0.0, 1e9)
    ent = {i: GateDrive("light_shift", (0, 1), light_shift=couplings) for i in (0, 1)}
    dev_ls = dataclasses.replace(dev, roles=dataclasses.replace(dev.roles, entangling=ent))
    with pytest.raises(ScheduleError, match="opposite sign"):
        schedule(Circuit(2, (Operation("zz", (0, 1), (math.pi / 2,)),), (0, 1)), dev_ls, ls_table)
    sch = schedule(Circuit(2, (Operation("zz", (0, 1), (-math.pi / 2,)),), (0, 1)), dev_ls, ls_table)
    ids = [p.gate_id or "" for p in sch.pulses]
    assert (
        sum("loop1" in i for i in ids) == 2
        and sum("loop2" in i for i in ids) == 2
        and sum("echo" in i for i in ids) == 4
    )
    with pytest.raises(ValueError):
        GateDrive("raman", (0, 1), light_shift=couplings)
    with pytest.raises(ValueError):
        LightShiftCouplings((-1.0, 0.0), 0.0, 1e9)


def test_ionq_json_ms_schedules_with_the_waveform(calibrated) -> None:  # type: ignore[no-untyped-def]
    dev, _modes, _drives, space, table, run = calibrated
    circ = load_ionq_json(
        {
            "gateset": "native",
            "qubits": 2,
            "circuit": [{"gate": "ms", "targets": [0, 1], "phases": [0.0, 0.25], "angle": 0.25}],
        }
    )
    sch = schedule(circ, dev, table)
    wf = table.waveform_for((0, 1))
    assert wf is not None and sch.pulses_end_s == pytest.approx(wf.duration_s + dev.hardware.dead_time_s)
    assert sch.measurement is not None and sch.duration_s == pytest.approx(
        sch.pulses_end_s + dev.detector.window_s
    ), "the IonQ circuit measures every qubit: the terminal event follows the pulses"
    eng = JointExactEngine(table=table)
    tr = eng.run_pulses(
        dev, sch, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    assert _fidelity(
        tr.final.internal, native_ms(0.0, math.pi / 2, math.pi / 2), KET00, sch.phase_frame
    ) > 1.0 - 3 * (1.0 - run.checks[-1].fidelity)


# ---- serialization under parallel addressing --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def four_ion():  # type: ignore[no-untyped-def]
    """A four-ion chain with one calibrated symmetric waveform served for both disjoint pairs (0, 1) and (2, 3)."""
    preset = yb171_chain(4)
    fx = dataclasses.replace(preset, device=dataclasses.replace(preset.device, preparation=None))
    waveforms = {}
    for pair in ((0, 1), (2, 3)):
        modes = gate_modes(fx.device, pair, (0, 1))
        waveforms[pair] = symmetric_pulse(
            modes, gate_mode=modes.modes[-1], loops=1, epsilon_hz=20e3, pair=pair
        ).waveform
    rabi, stark = derived_seeds(fx.device, fx.gate_drives)
    table = table_with_waveform((0, 1), waveforms[(0, 1)], rabi_hz=rabi, stark_hz=stark)
    table = dataclasses.replace(table, ms=waveforms)
    return fx, table


def _span(sch, gate_id_prefix: str) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    ps = [p for p in sch.pulses if (p.gate_id or "").startswith(gate_id_prefix)]
    assert ps, f"no pulses with gate id {gate_id_prefix!r}"
    return min(p.t_start_s for p in ps), max(p.t_end_s for p in ps)


@pytest.mark.parametrize("parallel_addressing", [False, True])
def test_two_ms_gates_on_disjoint_pairs_never_overlap(four_ion, parallel_addressing: bool) -> None:  # type: ignore[no-untyped-def]
    """Two MS gates on disjoint pairs run one after the other, separated by exactly the dead time, with or without parallel
    addressing."""
    fx, table = four_ion
    dev = dataclasses.replace(
        fx.device,
        hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=parallel_addressing),
    )
    circ = Circuit(
        4,
        (Operation("ms", (0, 1), (0.0, 0.0, MS)), Operation("ms", (2, 3), (0.0, 0.0, MS))),
        (0, 1, 2, 3),
    )
    sch = schedule(circ, dev, table)
    first, second = _span(sch, "ms[0]"), _span(sch, "ms[1]")
    assert first[1] <= second[0] + 1e-15, (
        f"parallel_addressing={parallel_addressing}: ms[0] {first} overlaps ms[1] {second}"
    )
    # they are separated by exactly the hardware dead time, not by more
    assert second[0] == pytest.approx(first[1] + dev.hardware.dead_time_s)
    assert len(sch.gates) == 2 and {g.pair for g in sch.gates} == {(0, 1), (2, 3)}
    assert sch.gates[0].t_end_s <= sch.gates[1].t_start_s + 1e-15
    # every entangling pulse of the two gates draws on the same global Raman pair, which is why they cannot share a slot
    assert {g.beams for g in sch.gates} == {(0, 1)}


def test_parallel_addressing_overlaps_single_qubit_gates_but_the_serial_default_does_not(four_ion) -> None:  # type: ignore[no-untyped-def]
    """Carrier pulses on distinct ions share one window when the hardware allows parallel addressing and are sequenced by the
    serial default."""
    fx, table = four_ion
    circ = Circuit(4, (Operation("gpi2", (0,), (0.0,)), Operation("gpi2", (2,), (0.0,))), (0, 1, 2, 3))
    serial = schedule(circ, fx.device, table)
    a, b = _span(serial, "gpi2[0]"), _span(serial, "gpi2[1]")
    assert a[1] <= b[0], "the serial default sequences single-qubit gates"
    dev_par = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    par = schedule(circ, dev_par, table)
    a2, b2 = _span(par, "gpi2[0]"), _span(par, "gpi2[1]")
    assert a2 == b2, "distinct addressing beams play the two carrier pulses in the same window"


def test_a_single_qubit_gate_after_an_ms_still_waits_for_it_under_parallel_addressing(four_ion) -> None:  # type: ignore[no-untyped-def]
    """Under parallel addressing a carrier pulse on a gate ion waits for the MS it follows while a spectator's starts at
    t = 0."""
    fx, table = four_ion
    dev = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    circ = Circuit(
        4,
        (
            Operation("ms", (0, 1), (0.0, 0.0, MS)),
            Operation("gpi2", (1,), (0.0,)),
            Operation("gpi2", (3,), (0.0,)),
        ),
        (0, 1, 2, 3),
    )
    sch = schedule(circ, dev, table)
    ms_span = _span(sch, "ms[0]")
    on_gate_ion = _span(sch, "gpi2[1]")
    spectator = _span(sch, "gpi2[2]")
    assert on_gate_ion[0] >= ms_span[1], "ion 1 took part in the MS gate"
    # ion 3's beam is free during the MS gate, so a parallel chain starts its carrier pulse at t0
    assert spectator[0] == pytest.approx(0.0)


def test_parallel_true_is_refused_when_the_device_model_does_not_allow_it(four_ion) -> None:  # type: ignore[no-untyped-def]
    """``parallel=True`` is refused on hardware without parallel addressing, and ``parallel=False`` is allowed on hardware
    with it."""
    fx, table = four_ion
    circ = Circuit(4, (Operation("gpi2", (0,), (0.0,)),), (0, 1, 2, 3))
    assert fx.device.hardware.parallel_addressing is False
    with pytest.raises(ScheduleError, match="parallel_addressing"):
        schedule(circ, fx.device, table, parallel=True)
    # explicitly serial on a parallel chain is allowed (the caller may want the serial path, e.g. crosstalk echoes)
    dev = dataclasses.replace(
        fx.device, hardware=dataclasses.replace(fx.device.hardware, parallel_addressing=True)
    )
    assert schedule(circ, dev, table, parallel=False).pulses
