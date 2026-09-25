"""The laboratory runs on machines: every experiment returns its typed result whose attributes equal ``value(key)``;
``requested`` and ``realized`` differ when the hardware chain quantises and agree otherwise; ``calibrate(machine)`` returns
the report."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from qutip_trap.benchmarks.budget import gate_channel, kind_of
from qutip_trap.benchmarks.rb import randomized_benchmarking
from qutip_trap.calibration import calibrate
from qutip_trap.calibration.experiments import CalibrationReport
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.device.presets import ideal_hardware
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.experiments.imaging import crystal_image
from qutip_trap.experiments.motion import thermometry
from qutip_trap.experiments.readout import detection_histogram
from qutip_trap.experiments.result import (
    RESULT_TYPES,
    CrystalImage,
    DetectionHistogram,
    ExperimentResult,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    ThermometryResult,
)
from qutip_trap.experiments.single_ion import rabi_scan, ramsey, sideband_spectroscopy
from qutip_trap.machine import Machine, laboratory_kwargs
from qutip_trap.options import Numerics, Truncation
from tests.m6_fixtures import CircuitFixture, circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = Numerics(truncation=Truncation(branch_weight_min=1e-3))
DURATIONS = np.linspace(0.0, 20e-6, 9)
DETUNINGS = np.linspace(-3.2e6, 3.2e6, 9)


@pytest.fixture(scope="module")
def fx() -> CircuitFixture:
    return circuit_fixture(2)


@pytest.fixture(scope="module")
def machine(fx: CircuitFixture) -> Machine:
    return Machine(fx.device, numerics=FAST).calibrated(
        pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS
    )


EXPERIMENTS: dict[str, Any] = {
    "rabi_scan": lambda m, **kw: rabi_scan(m, 0, DURATIONS, shots=200, seed=1, **kw),
    "ramsey": lambda m, **kw: ramsey(
        m, 0, np.linspace(0.0, 40e-6, 6), shots=200, seed=1, detuning_hz=2e4, **kw
    ),
    "sideband_spectroscopy": lambda m, **kw: sideband_spectroscopy(m, 0, DETUNINGS, seed=1, **kw),
    "thermometry": lambda m, **kw: thermometry(m, 0, coupled_mode(m), shots=200, seed=1, **kw),
}


def coupled_mode(machine: Machine | Any) -> int:
    """The mode ion 0's gate drive couples to most strongly (the fixture's x modes; mode 0 is uncoupled)."""
    from qutip_trap.light.raman import derive_raman_drive

    device = machine.device if isinstance(machine, Machine) else machine
    beams = device.roles.resolve(device).gate[0].beams
    etas = derive_raman_drive(device, 0, (beams[0], beams[1]), scattering=False).etas
    return max(etas, key=lambda m: abs(etas[m]))


KINDS = {
    "rabi_scan": RabiScan,
    "ramsey": RamseyFringe,
    "sideband_spectroscopy": SidebandSpectrum,
    "thermometry": ThermometryResult,
}


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_an_experiment_on_the_machine_returns_its_typed_result(name: str, fx: CircuitFixture) -> None:
    on_machine = EXPERIMENTS[name](Machine(fx.device))
    assert isinstance(on_machine, KINDS[name]) and RESULT_TYPES[name] is KINDS[name]
    # every typed attribute equals value(key) where the fit produced the key
    for key in on_machine.fitted:
        attr = key if key.isidentifier() else None
        if attr is not None and isinstance(getattr(type(on_machine), attr, None), property):
            assert getattr(on_machine, attr) == on_machine.value(key), key
    assert on_machine.quality in ("good", "poor", "exact", "failed") and on_machine.created_at


def test_the_result_table_names_every_experiment_of_the_calibration_order() -> None:
    from qutip_trap.calibration.experiments import ALIASES, ORDER

    assert set(RESULT_TYPES) >= set(ORDER) | {"ramsey", "thermometry", "mode_spectroscopy", "ms_phase_scan"}
    assert {a for a in ALIASES if a in RESULT_TYPES} == {"thermometry", "mode_spectroscopy", "ms_phase_scan"}
    for cls in RESULT_TYPES.values():
        assert issubclass(cls, ExperimentResult) and cls.x_label and cls.y_label


def test_the_machine_supplies_table_options_and_builder_as_defaults(machine: Machine) -> None:
    device, kw = laboratory_kwargs(machine, {"shots": 5})
    assert device is machine.device and kw["shots"] == 5
    assert kw["table"] is machine.table and kw["options"] == machine.numerics.to_solver_options(
        machine.physics
    )
    assert "builder_options" not in kw  # the machine's builder is None: nothing to supply
    _device, given = laboratory_kwargs(machine, {"table": None, "options": SolverOptions(atol=1e-12)})
    assert given["table"] is None and given["options"].atol == 1e-12  # the call's own keywords win


# ---- 2.3: requested and realized -------------------------------------------------------------------------------------------------


def test_realized_equals_requested_on_ideal_hardware_and_moves_when_the_chain_quantises(
    fx: CircuitFixture,
) -> None:
    ideal = rabi_scan(Machine(fx.device), 0, DURATIONS, seed=1)
    assert ideal.requested is not None and ideal.realized is not None
    assert ideal.requested.durations_s == ideal.realized.durations_s == tuple(float(t) for t in DURATIONS)
    assert ideal.requested.rabi_hz == ideal.realized.rabi_hz and ideal.requested.phase_rad == (0.0,)
    assert ideal.requested.durations_s[3] == pytest.approx(DURATIONS[3])
    coarse = dataclasses.replace(
        fx.device,
        hardware=dataclasses.replace(
            ideal_hardware(),
            dds_amplitude_bits=6,
            amplitude_full_scale_hz=1e6,
            dds_clock_hz=1e9,
            dds_frequency_bits=20,
        ),
    )
    quantised = rabi_scan(Machine(coarse), 0, DURATIONS, seed=1)
    assert quantised.realized is not None and quantised.requested is not None
    assert quantised.requested.rabi_hz == ideal.requested.rabi_hz  # the request does not know the electronics
    assert quantised.realized.rabi_hz != quantised.requested.rabi_hz  # the amplitude word does
    assert abs(quantised.realized.rabi_hz[0] - quantised.requested.rabi_hz[0]) <= 1e6 / 2**6
    spectrum = sideband_spectroscopy(Machine(coarse), 0, DETUNINGS, seed=1)
    assert spectrum.realized is not None and spectrum.requested is not None
    grid = 1e9 / 2**20
    for want, got in zip(spectrum.requested.detunings_hz, spectrum.realized.detunings_hz):
        assert abs(got - want) <= 0.5 * grid + 1e-9 and (
            want == got or got % grid == pytest.approx(0.0, abs=1e-6)
        )
    assert spectrum.realized.detunings_hz != spectrum.requested.detunings_hz


def test_scan_parameters_are_tuples_by_name() -> None:
    p = ScanParameters({"delays_s": np.array([1e-6, 2e-6]), "rabi_hz": 1e5})
    assert p.delays_s == (1e-6, 2e-6) and p.rabi_hz == (1e5,) and p.keys() == ("delays_s", "rabi_hz")
    with pytest.raises(AttributeError, match="no parameter 'phase_rad'"):
        _ = p.phase_rad
    assert p == ScanParameters({"delays_s": [1e-6, 2e-6], "rabi_hz": (1e5,)}) and p.as_dict()["rabi_hz"] == (
        1e5,
    )


def test_detection_histogram_and_crystal_image_are_typed(machine: Machine) -> None:
    hist = detection_histogram(machine, 0, 200, seed=2)
    assert isinstance(hist, DetectionHistogram) and hist.threshold is not None and hist.window_s is not None
    assert hist.errors is not None and hist.errors == (hist.eps_B, hist.eps_D) and hist.subject == {"ion": 0}
    assert hist.requested is not None and hist.requested.n_records == (200.0,) and hist.realized is None
    image = crystal_image(machine, seed=2)
    assert (
        isinstance(image, CrystalImage)
        and image.n_ions == 2.0
        and image.quality in ("good", "exact", "failed")
    )


# ---- 2.2: calibrate on machines, the deprecated names ------------------------------------------------------------------------------


def test_calibrate_returns_the_report(fx: CircuitFixture, machine: Machine) -> None:
    scans: dict[str, Any] = {"pairs": [(0, 1)], "detection_records": 300, "detection_windows_s": WINDOWS}
    report = calibrate(Machine(fx.device, numerics=FAST), **scans)
    assert (
        isinstance(report, CalibrationReport) and report.table == machine.table and report.experiments == ()
    )
    assert report.results == {} and report.refused == {} and report.surrogate.table == report.table
    assert report.sample.sample_id == 0
    assert all(v == 0.0 for v in report.surrogate_error().values())  # the surrogate against itself
    with pytest.raises(ValueError, match="closed_form"):
        calibrate(machine, method="guess")  # type: ignore[arg-type]
    # the machine's t0_s is the calibration's default time
    late = dataclasses.replace(machine, physics=dataclasses.replace(machine.physics, t0_s=3.0))
    assert calibrate(late, **scans).table.fitted_at_s == pytest.approx(3.0)


# ---- 2.2: the benchmarks on machines -------------------------------------------------------------------------------------------------


def test_benchmarks_take_a_machine(machine: Machine) -> None:
    kind = kind_of("gpi2", (0,))
    ch = gate_channel(machine, kind)
    assert ch.kind == kind and ch.level == "GATE_LOCAL" and gate_channel(machine, kind) is ch
    with pytest.raises(TypeError, match="unexpected keyword"):
        randomized_benchmarking(machine, (0,), (1,), n_sequences=1, shots=5, budget=False, shotz=1)
    with pytest.raises(ValueError, match="pair=True"):
        randomized_benchmarking(machine, (0,), (1,), n_sequences=1, shots=5, budget=False, pair=True)
