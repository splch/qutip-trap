"""ms and zz through the scheduler (PLAN.md Sections 4.4.2, 7.1, 7.3, 7.6, 12; Appendix E): the native matrices, the sign and axis
conventions, the ZZ wrapper construction, virtual-Z frames, partial angles, refusals and the IonQ JSON path."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.api import Circuit, Operation, SeedSpec, SolverOptions, load_ionq_json
from qutip_trap.calibration.entangling import calibrate_entangling_angle, gate_space
from qutip_trap.control.native import equal_up_to_global_phase, gpi2, rz, xx
from qutip_trap.control.native import ms as native_ms
from qutip_trap.control.native import zz as native_zz
from qutip_trap.control.schedule import GateDrive, ScheduleError, ms_spin_phases, schedule
from qutip_trap.control.shaping import solve_amplitude_modulation
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.noise.sampling import quiet_sample
from tests.m4_fixtures import (
    X_COM_TWO_IONS,
    raman_gate_drives,
    table_with_waveform,
    two_ion_device,
    two_ion_modes,
)

RABI_TABLE = {(0, 0): 100e3, (1, 0): 100e3}


@pytest.fixture(scope="module")
def calibrated():  # type: ignore[no-untyped-def]
    """The two-mode AM gate, exactly calibrated to chi = pi/4, with its table, space and drives."""
    dev = two_ion_device()
    modes = two_ion_modes(dev)
    drives = raman_gate_drives(2)
    am = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    space = gate_space(modes, 2, waveform=am.waveform)
    table0 = table_with_waveform((0, 1), am.waveform, rabi_hz=RABI_TABLE)
    run = calibrate_entangling_angle(
        dev, am.waveform, (0, 1), drives, table0, space=space, tolerance_rad=2e-4
    )
    assert run.converged
    table = table_with_waveform((0, 1), run.waveform, rabi_hz=RABI_TABLE)
    return dev, modes, drives, space, table, run


def _run_circuit(dev, table, space, ops, internal=(0, 0), **kw):  # type: ignore[no-untyped-def]
    circ = Circuit(2, tuple(ops), (0, 1))
    sch = schedule(circ, dev, table, **kw)
    eng = JointExactEngine()
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


def _fidelity(rho: qt.Qobj, mat: np.ndarray, ket0: np.ndarray) -> float:
    target = qt.Qobj(mat @ ket0)
    target.dims = [[2, 2], [1, 1]]
    return float(np.real(qt.expect(rho, target)))


KET00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def test_ms_reproduces_the_native_matrix_for_arbitrary_phases(calibrated) -> None:  # type: ignore[no-untyped-def]
    """MS(phi_0, phi_1, pi/2) = exp[-i (pi/4) GPi(phi_0) (x) GPi(phi_1)] on |00> to the gate's exact infidelity (Section 7.6)."""
    dev, _modes, _drives, space, table, run = calibrated
    budget = 1.0 - run.checks[-1].fidelity
    assert budget < 1e-3
    for phi0, phi1 in ((0.0, 0.0), (0.3, 1.1), (-0.7, 2.0)):
        rho, sch = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (phi0, phi1, math.pi / 2))])
        assert _fidelity(rho, native_ms(phi0, phi1, math.pi / 2), KET00) > 1.0 - 3 * budget
        assert _fidelity(rho, native_ms(phi0, phi1, -math.pi / 2), KET00) < 0.02
        assert len(sch.pulses) == 10 and all(p.closes_modes == (2, 3) for p in sch.pulses)
    # a positive kernel sign is played with pi on the second ion (Section 13: exp(+i chi sigma sigma) = XX(-chi))
    wf = table.waveform_for((0, 1))
    assert wf is not None and wf.chi_total_rad > 0.0
    spins, chi_abs = ms_spin_phases(wf, (0, 1), (0.0, 0.0), PhaseFrame())
    assert spins[0] == pytest.approx(-math.pi / 2) and spins[1] == pytest.approx(math.pi / 2)
    assert chi_abs == pytest.approx(wf.chi_total_rad)


def test_partial_angle_rescales_by_the_s_squared_law(calibrated) -> None:  # type: ignore[no-untyped-def]
    """MS(0, 0, theta) at theta = 0.6: every amplitude scales by sqrt((theta/2)/(pi/4)) and P_11 = sin^2(theta/2) (Section 4.4.7 (7))."""
    dev, _modes, _drives, space, table, _run = calibrated
    theta = 0.6
    rho, sch = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (0.0, 0.0, theta))])
    wf = table.waveform_for((0, 1))
    assert wf is not None and wf.segments is not None
    played = sch.pulses[0].drive.tones[0].envelope_hz
    assert abs(wf.chi_total_rad - math.pi / 4) < 2e-4, "the calibrated waveform carries its exact angle"
    amp0 = wf.segments[0].amplitude_hz[(0, "blue")]
    assert not callable(amp0) and not callable(played)
    expected = float(amp0) * math.sqrt((theta / 2) / abs(wf.chi_total_rad))
    assert float(played) == pytest.approx(expected, rel=1e-12)
    p11 = float(np.real(rho.full()[3, 3]))
    assert p11 == pytest.approx(math.sin(theta / 2) ** 2, abs=3e-3)
    assert _fidelity(rho, native_ms(0.0, 0.0, theta), KET00) > 0.995
    # a negative angle is a pi on the second phase
    rho_n, _ = _run_circuit(dev, table, space, [Operation("ms", (0, 1), (0.0, 0.0, -theta))])
    assert _fidelity(rho_n, native_ms(0.0, 0.0, -theta), KET00) > 0.995


def test_zz_wrapper_construction_matrix_and_schedule(calibrated) -> None:  # type: ignore[no-untyped-def]
    """ZZ(theta) = [GPi2(pi/2) (x) GPi2(pi/2)] XX(theta/2) [GPi2(3 pi/2) (x) GPi2(3 pi/2)] exactly (R_y(pi/2) X R_y(pi/2)^dag = -Z), the
    inferred construction of Section 12; the schedule plays wrappers, the MS segments and wrappers with dead times between."""
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
    assert _fidelity(rho, native_zz(math.pi / 2), ket) > 1.0 - 3 * budget - 5e-3
    assert _fidelity(rho, native_zz(-math.pi / 2), ket) < 0.05
    # the wrappers are 2.5 us GPi2 pulses at the table's 100 kHz with the dead time before and after the MS block
    wrap = [p for p in sch.pulses if "wrap_in" in (p.gate_id or "")]
    assert all(p.duration_s == pytest.approx(0.25 / 100e3) for p in wrap)
    ms_start = min(p.t_start_s for p in sch.pulses if "/ms/" in (p.gate_id or ""))
    assert ms_start == pytest.approx(max(p.t_end_s for p in wrap) + dev.hardware.dead_time_s)


def test_virtual_z_frame_carries_through_ms(calibrated) -> None:  # type: ignore[no-untyped-def]
    """RZ(theta) on ion 1 then MS(0, 0): the played state is RZ_1(theta)^dag [MS(0, 0) RZ_1(theta)] |00> (the frame the measurement
    discards, Sections 5.2 and 7.6), equivalently MS(0, -theta) |00>; comparing with MS(0,0) RZ|00> directly fails by cos^2(theta/2)."""
    dev, _modes, _drives, space, table, run = calibrated
    theta = 0.4
    rho, sch = _run_circuit(
        dev, table, space, [Operation("rz", (1,), (theta,)), Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2))]
    )
    assert sch.phase_frame == {0: 0.0, 1: pytest.approx(theta)}
    budget = 1.0 - run.checks[-1].fidelity
    assert _fidelity(rho, native_ms(0.0, -theta, math.pi / 2), KET00) > 1.0 - 3 * budget
    circuit_order = native_ms(0.0, 0.0, math.pi / 2) @ np.kron(np.eye(2), rz(theta))
    assert _fidelity(rho, np.kron(np.eye(2), rz(theta)).conj().T @ circuit_order, KET00) > 1.0 - 3 * budget
    assert _fidelity(rho, circuit_order, KET00) == pytest.approx(math.cos(theta / 2) ** 2, abs=3e-3)


def test_scheduler_refusals_and_table_lookup(calibrated) -> None:  # type: ignore[no-untyped-def]
    dev, modes, drives, _space, table, _run = calibrated
    empty = table_with_waveform((0, 1), Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=20e3))
    import dataclasses

    none = dataclasses.replace(empty, ms={})
    with pytest.raises(ScheduleError, match="no entangling waveform"):
        schedule(Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1)), dev, none)
    # the reversed key is found
    assert dataclasses.replace(table, ms={(1, 0): table.ms[(0, 1)]}).waveform_for((0, 1)) is not None
    # a light-shift waveform cannot serve an MS gate, and needs a light-shift drive
    ls = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=20e3, kind="light_shift")
    ls_table = table_with_waveform((0, 1), ls, rabi_hz=RABI_TABLE)
    with pytest.raises(ScheduleError, match="MS .* waveform"):
        schedule(Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1)), dev, ls_table)
    with pytest.raises(ScheduleError, match="light_shift gate drive"):
        schedule(Circuit(2, (Operation("zz", (0, 1), (-math.pi / 2,)),), (0, 1)), dev, ls_table)
    # a sigma_z force takes its sign from the detuning side only: the wrong sign is refused, not phase-flipped
    from qutip_trap.control.pulses import LightShiftCouplings

    couplings = LightShiftCouplings((-2.0, 0.0), 0.0, 1e9)
    ent = {i: GateDrive("light_shift", (0, 1), light_shift=couplings) for i in (0, 1)}
    with pytest.raises(ScheduleError, match="opposite sign"):
        schedule(
            Circuit(2, (Operation("zz", (0, 1), (math.pi / 2,)),), (0, 1)),
            dev,
            ls_table,
            entangling_drives=ent,
        )
    sch = schedule(
        Circuit(2, (Operation("zz", (0, 1), (-math.pi / 2,)),), (0, 1)), dev, ls_table, entangling_drives=ent
    )
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
    assert wf is not None and sch.duration_s == pytest.approx(wf.duration_s + dev.hardware.dead_time_s)
    eng = JointExactEngine()
    tr = eng.run_pulses(
        dev, sch, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    assert _fidelity(tr.final.internal, native_ms(0.0, math.pi / 2, math.pi / 2), KET00) > 1.0 - 3 * (
        1.0 - run.checks[-1].fidelity
    )
