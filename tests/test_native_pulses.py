"""Single-qubit native gates as pulses and the virtual-Z frame (PLAN.md Section 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.native import equal_up_to_global_phase, gpi, gpi2, rz
from qutip_trap.control.schedule import (
    MICROWAVE_BEAM_KEY,
    GateDrive,
    ScheduleError,
    default_gate_drives,
    schedule,
    single_qubit_pulse,
)
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive
from qutip_trap.noise.sampling import quiet_sample
from tests.m2_fixtures import microwave_device, single_ion_raman_device, table_with_rabi

KET0 = np.array([1.0, 0.0])
RABI_HZ = 20661.157  # a 12.1 us pi/2 pulse (Harty)


def _run_microwave(ops: list[Operation]) -> tuple[np.ndarray, object]:
    dev = microwave_device()
    table = table_with_rabi({(0, MICROWAVE_BEAM_KEY): RABI_HZ})
    circ = Circuit(1, tuple(ops), (0,))
    sch = schedule(circ, dev, table)
    space = HilbertSpace((2,), (), None, (0, 1, 2))
    eng = JointExactEngine()
    tr = eng.run_pulses(
        dev, sch, space.initial_state([0]), space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    assert tr.final.joint is not None
    return tr.final.joint.full().ravel(), sch


def test_gpi2_and_gpi_pulses_reproduce_the_native_matrices_at_eta_zero() -> None:
    """GPi2(0)|0> = (|0> - i|1>)/sqrt2; a microwave carrier (eta = 0) reproduces the native matrices to 1e-6."""
    psi, sch = _run_microwave([Operation("gpi2", (0,), (0.0,))])
    assert equal_up_to_global_phase(psi.reshape(2, 1), (gpi2(0.0) @ KET0).reshape(2, 1), atol=1e-6)
    assert np.allclose(np.abs(psi) ** 2, [0.5, 0.5], atol=1e-6)
    assert np.angle(psi[1] / psi[0]) == pytest.approx(-math.pi / 2, abs=1e-6), (
        "GPi2(0)|0> = (|0> - i|1>)/sqrt2"
    )
    assert len(sch.pulses) == 1 and sch.pulses[0].duration_s == pytest.approx(0.25 / RABI_HZ)
    assert sch.idle == ((sch.pulses[0].t_end_s, sch.pulses[0].t_end_s + 1e-6),), (
        "the hardware dead time follows every pulse"
    )
    for phi in (0.3, 2.0, -1.1):
        psi_g, _ = _run_microwave([Operation("gpi", (0,), (phi,))])
        assert equal_up_to_global_phase(psi_g.reshape(2, 1), (gpi(phi) @ KET0).reshape(2, 1), atol=1e-6)
        psi_h, _ = _run_microwave([Operation("gpi2", (0,), (phi,))])
        assert equal_up_to_global_phase(psi_h.reshape(2, 1), (gpi2(phi) @ KET0).reshape(2, 1), atol=1e-6)
    # a two-pulse sequence: GPi2(0.4) then GPi(1.0) equals the time-ordered matrix product (right to left)
    psi2, _ = _run_microwave([Operation("gpi2", (0,), (0.4,)), Operation("gpi", (0,), (1.0,))])
    assert equal_up_to_global_phase(
        psi2.reshape(2, 1), (gpi(1.0) @ gpi2(0.4) @ KET0).reshape(2, 1), atol=1e-6
    )


def test_virtual_z_concrete_sequence_pins_the_sign() -> None:
    """RZ(0.1) then GPi2(0) on |0> equals GPi2(-0.1) followed by RZ(0.1) (phi -> phi - theta), and the opposite sign does
    not."""
    psi, sch = _run_microwave([Operation("rz", (0,), (0.1,)), Operation("gpi2", (0,), (0.0,))])
    assert sch.pulses[0].drive.tones[0].phase_rad == pytest.approx(-0.1)
    assert sch.phase_frame == {0: pytest.approx(0.1)}
    matrices = gpi2(0.0) @ rz(0.1) @ KET0
    assert np.allclose(matrices, rz(0.1) @ gpi2(-0.1) @ KET0)
    assert equal_up_to_global_phase(psi.reshape(2, 1), (gpi2(-0.1) @ KET0).reshape(2, 1), atol=1e-6)
    assert equal_up_to_global_phase((rz(0.1) @ psi).reshape(2, 1), matrices.reshape(2, 1), atol=1e-6)
    assert not equal_up_to_global_phase(psi.reshape(2, 1), (gpi2(+0.1) @ KET0).reshape(2, 1), atol=1e-3)
    frame = PhaseFrame().rz(0, 0.1).rz(0, 0.2)
    assert frame.pulse_phase(0, 1.0) == pytest.approx(0.7) and frame.pulse_phase(1, 1.0) == 1.0


def test_scheduler_refusals_and_drive_inference() -> None:
    dev = microwave_device()
    assert default_gate_drives(dev) == {0: GateDrive("microwave", ())}
    raman = single_ion_raman_device()
    assert default_gate_drives(raman) == {0: GateDrive("raman", (0, 1))}
    circ = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
    with pytest.raises(ScheduleError, match="no carrier Rabi frequency"):
        schedule(circ, dev, table_with_rabi({}))
    with pytest.raises(ScheduleError, match="uncalibrated"):
        schedule(circ, dev, table_with_rabi({(0, MICROWAVE_BEAM_KEY): RABI_HZ}, status="uncalibrated"))
    table = table_with_rabi({(0, MICROWAVE_BEAM_KEY): RABI_HZ, (1, MICROWAVE_BEAM_KEY): RABI_HZ})
    with pytest.raises(ScheduleError, match="no entangling waveform"):
        schedule(Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2)),), (0, 1)), dev, table)
    # a measure or reset before a later gate is refused
    with pytest.raises(ScheduleError, match="mid-circuit"):
        schedule(
            Circuit(1, (Operation("measure", (0,), ()), Operation("gpi", (0,), (0.0,))), (0,)), dev, table
        )
    with pytest.raises(ScheduleError, match="mid-circuit"):
        schedule(Circuit(1, (Operation("reset", (0,), ()), Operation("gpi", (0,), (0.0,))), (0,)), dev, table)
    with pytest.raises(ScheduleError, match="native"):
        schedule(Circuit(1, (Operation("h", (0,), ()),), (0,)), dev, table)
    # the terminal measure is the schedule's one event, after the last pulse and dead time, of the detector's window when the
    # table carries no detection entry; two gates run sequentially with a dead time between them
    sch = schedule(
        Circuit(
            1,
            (Operation("gpi", (0,), (0.0,)), Operation("gpi2", (0,), (0.5,)), Operation("measure", (0,), ())),
            (0,),
        ),
        dev,
        table,
    )
    assert len(sch.pulses) == 2 and len(sch.events) == 1
    meas = sch.measurement
    assert meas is not None and meas.kind == "measure" and meas.ions == (0,)
    assert meas.t_start_s == pytest.approx(sch.pulses[1].t_end_s + dev.hardware.dead_time_s)
    assert meas.t_end_s - meas.t_start_s == pytest.approx(dev.detector.window_s)
    assert sch.pulses_end_s == pytest.approx(meas.t_start_s) and sch.duration_s == pytest.approx(meas.t_end_s)
    assert schedule(Circuit(1, (Operation("gpi", (0,), (0.0,)),), ()), dev, table).events == ()
    assert sch.pulses[1].t_start_s == pytest.approx(sch.pulses[0].t_end_s + dev.hardware.dead_time_s)
    assert sch.pulses[0].duration_s == pytest.approx(2 * sch.pulses[1].duration_s)
    pulse = single_qubit_pulse(0, math.pi, 0.2, GateDrive("microwave", ()), RABI_HZ, 1e-3, gate_id="x")
    assert pulse.duration_s == pytest.approx(0.5 / RABI_HZ) and pulse.t_start_s == 1e-3
    with pytest.raises(ValueError):
        single_qubit_pulse(0, -math.pi, 0.0, GateDrive("microwave", ()), RABI_HZ, 0.0)


def test_raman_gpi2_reproduces_the_matrix_within_the_debye_waller_budget() -> None:
    """With motion resolved the identity holds within the intrinsic budget (off-resonant carrier (Omega/nu)^2, the
    Debye-Waller factor); the table's Rabi frequency is the DW-reduced carrier the calibration would have measured."""
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    eta = dd.etas[1]
    table = table_with_rabi({(0, 0): dd.carrier_rabi_hz * math.exp(-(eta**2) / 2)})
    sch = schedule(Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)), dev, table)
    space = HilbertSpace((2,), (ModeTruncation(1, 10, (0, 2), 0.2),), None, (0, 2))
    eng = JointExactEngine()
    tr = eng.run_pulses(
        dev, sch, space.initial_state([0], fock={1: 0}), space, quiet_sample(), SeedSpec(0), SolverOptions()
    )
    rho = tr.final.internal.full()
    target = gpi2(0.0) @ KET0
    fidelity = float(np.real(target.conj() @ rho @ target))
    budget = (dd.carrier_rabi_hz / 3.0e6) ** 2 + eta**2 * 0.01
    assert 1.0 - fidelity < 5 * budget and 1.0 - fidelity < 1e-3
