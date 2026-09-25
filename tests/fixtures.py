"""Shared fixtures: a small but complete Appendix E ``Device`` and the records ``Result`` needs.

Numbers here are FIXTURE values for exercising the data model, not physics results; the two-ion 171Yb+ mode
frequencies follow the realizable Section 11.1 fixture (x-COM 3.000, x-rocking 2.828, y-COM 2.900, axial 1.000 MHz).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from qutip_trap.control.compiler import Circuit
from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.table import CalEntry, CalibrationTable
from qutip_trap.device.model import Device, Field
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.beams import Beam
from qutip_trap.machine import Machine
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.spectra import Drift, NoiseSpectrum
from qutip_trap.readout.detection import Detector
from qutip_trap.run.results import Diagnostics, Progress, Result, RunState
from qutip_trap.species import species
from qutip_trap.trap.crystal import Crystal, Mode
from qutip_trap.trap.model import Trap

SQ = 1.0 / math.sqrt(2.0)


def make_crystal() -> Crystal:
    yb = species("171Yb+")
    com = np.array([SQ, SQ])
    rock = np.array([-SQ, SQ])
    modes = (
        Mode("axial", 0, 1.000e6, (0.0, 0.0, 1.0), com),
        Mode("axial", 1, math.sqrt(3.0) * 1.000e6, (0.0, 0.0, 1.0), rock),
        Mode("transverse_1", 0, 2.828e6, (1.0, 0.0, 0.0), rock),
        Mode("transverse_1", 1, 3.000e6, (1.0, 0.0, 0.0), com),
        Mode("transverse_2", 0, 2.720e6, (0.0, 1.0, 0.0), rock),
        Mode("transverse_2", 1, 2.900e6, (0.0, 1.0, 0.0), com),
    )
    positions = np.array([[0.0, 0.0, -2.7e-6], [0.0, 0.0, 2.7e-6]])
    return Crystal(species=(yb, yb), positions_m=positions, modes=modes)


def make_trap() -> Trap:
    return Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={"shim_x": 0.0, "shim_y": 0.0},
    )


def make_field() -> Field:
    return Field(B_gauss=5.0, direction=(0.0, 0.0, 1.0), noise=None)


def make_noise(s_e_two_sided: float = 0.0, correlation_length_m: float = 0.0) -> NoiseModel:
    """A quiet noise model by default (M7: run() assembles the device's channels, so the shared fixture carries no heating unless
    asked); ``s_e_two_sided`` sets a flat electric-field density in (V/m)^2/(rad/s) with the given correlation length."""
    omega = np.linspace(-2.0 * math.pi * 1e7, 2.0 * math.pi * 1e7, 5)
    flat = NoiseSpectrum(omega_rad_s=omega, S=np.full(5, s_e_two_sided), unit="(V/m)^2/(rad/s)")
    quiet = Drift(rms=0.0, tau_s=1.0, servo_bandwidth_hz=None)
    return NoiseModel(
        S_E=flat,
        correlation_length_m=correlation_length_m,
        S_B=None,
        mains=None,
        laser_phase=None,
        laser_intensity=None,
        rf_amplitude_noise=None,
        rf_phase_noise=None,
        rf_amplitude_drift=quiet,
        mode_drift_differential=quiet,
        rabi_drift=quiet,
        beam_phase_drift=quiet,
        field_drift=quiet,
        stray_field_drift=quiet,
        pointing_drift=quiet,
        rabi_amplitude=None,
        collisions=None,
    )


def make_detector() -> Detector:
    return Detector(
        kind="pmt",
        efficiency=0.02,
        background_cps=100.0,
        psf_leakage={},
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=100e-6,
    )


def make_hardware(realistic: bool = False) -> HardwareChain:
    """Near-ideal electronics by default (32-bit phase and 24-bit amplitude words, zero modulator rise time, infinite amplifier
    bandwidth), so that the
    exactness tests of M2 to M6 see the pulses they specify (Section 7.10: a device without these parameters gets ideal
    electronics and says so); ``realistic=True`` is the 16-bit / 14-bit / 50 ns chain the M7 hardware tests exercise."""
    if realistic:
        return HardwareChain(
            dds_phase_bits=16,
            dds_amplitude_bits=14,
            aom_rise_s=50e-9,
            amplifier_bandwidth_hz=1e8,
            dead_time_s=1e-6,
            phase_continuous=True,
        )
    return HardwareChain(
        dds_phase_bits=32,
        dds_amplitude_bits=24,
        aom_rise_s=0.0,
        amplifier_bandwidth_hz=float("inf"),
        dead_time_s=1e-6,
        phase_continuous=True,
    )


def make_raman_pair() -> tuple[Beam, Beam]:
    """Two 355 nm Raman beams crossing at 90 degrees, both polarized along z (the quantization axis)."""
    b1 = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    return b1, b2


def make_device() -> Device:
    return Device(
        crystal=make_crystal(),
        trap=make_trap(),
        field=make_field(),
        beams=make_raman_pair(),
        noise=make_noise(),
        detector=make_detector(),
        hardware=make_hardware(),
    )


def make_run_state() -> RunState:
    return RunState(order=(0, 1), dark=frozenset(), lost=frozenset(), events=())


def make_calibration_table() -> CalibrationTable:
    entry = CalEntry(
        value=5.0,
        uncertainty=0.001,
        status="seed",
        experiment="field_scan",
        provenance_id="conv.curvature_naming",
        fitted_at_s=0.0,
        sample_id=0,
    )
    return CalibrationTable(
        device_hash="fixture",
        seed=0,
        surrogate=True,
        qubit_freq={},
        rabi={},
        stark={},
        crosstalk={},
        modes={},
        nbar={},
        ms={},
        field=entry,
        micromotion={},
        detection={},
        heating={},
    )


def make_space() -> HilbertSpace:
    return HilbertSpace(
        ion_dims=(2, 2),
        resolved=(ModeTruncation(mode=3, d=8, expected_n_range=(0, 3), eta_max=0.1),),
        enr_group=None,
        frozen=(0, 1, 2, 4, 5),
    )


def make_diagnostics() -> Diagnostics:
    return Diagnostics(
        level="JOINT_EXACT",
        space=make_space(),
        mode_class={3: "resolved", 0: "frozen", 1: "frozen", 2: "frozen", 4: "frozen", 5: "frozen"},
        run_state=make_run_state(),
        wall_clock_span_s=0.0,
        boundary_population={3: 0.0},
        margin_levels={3: 0},
        dropped_modes=(),
        frozen_contribution={},
        integrator="dop853",
        tolerances=(1e-10, 1e-8),
        samples=1,
        trajectories=1,
        shots_per_sample=1,
        effective_sample_size=1.0,
        root_seed=0,
        calibration=make_calibration_table(),
        approximations=(),
    )


def make_result(bitstrings: np.ndarray) -> Result:
    from qutip_trap.run.results import aggregate

    counts, probabilities = aggregate(bitstrings)
    return Result(
        bitstrings=np.asarray(bitstrings, dtype=np.uint8),
        bit_order="qubit0_lsb",
        counts=counts,
        probabilities=probabilities,
        error_bars={k: 0.0 for k in counts},
        photon_records=None,
        posteriors=None,
        noise_samples=(),
        heralds=np.zeros(len(bitstrings), dtype=np.uint8),
        discarded_shots=0,
        run_state=make_run_state(),
        spam={},
        final_state=None,
        diagnostics=make_diagnostics(),
    )


def run(
    circuit: Circuit,
    device: Device,
    shots: int,
    *,
    seed: int = 0,
    keep_final_state: bool = False,
    progress: Callable[[Progress], None] | None = None,
    **machine: Any,
) -> Result:
    """``Machine(device, **machine).run(circuit, shots, ...)``: a run on a bare device with the machine's fields as keywords."""
    return Machine(device, **machine).run(
        circuit, shots, seed=seed, keep_final_state=keep_final_state, progress=progress
    )
