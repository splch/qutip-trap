"""docs/api_implementation_plan.md 2.2 and 2.3: the laboratory runs on machines. Every experiment, the calibration and the
benchmarks accept a ``Machine`` (a ``Device`` is wrapped in a default machine); the drive keywords are deprecated in favour
of the device's roles; every experiment returns its typed subclass whose attributes equal ``value(key)``; ``requested`` and
``realized`` differ when the hardware chain quantises and agree otherwise; ``calibrate(machine, method=)`` returns the report
and ``calibrate(device)`` the table; ``calibrate_with_report`` and ``compile_with_report`` warn."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest

from qutip_trap._compat import QutipTrapDeprecationWarning
from qutip_trap.api import Circuit, Operation, SolverOptions
from qutip_trap.benchmarks import gate_channel, randomized_benchmarking
from qutip_trap.benchmarks.budget import kind_of
from qutip_trap.calibration import CalibrationReport, calibrate, calibrate_with_report
from qutip_trap.control.compiler import compile_report, compile_with_report
from qutip_trap.control.table import CalibrationTable
from qutip_trap.device.presets import ideal_hardware
from qutip_trap.experiments import (
    RESULT_TYPES,
    CrystalImage,
    DetectionHistogram,
    ExperimentResult,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    ThermometryResult,
    crystal_image,
    detection_histogram,
    rabi_scan,
    ramsey,
    sideband_spectroscopy,
    thermometry,
)
from qutip_trap.machine import DRIVE_KEYWORDS, Machine, as_machine, laboratory_kwargs
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


def same_result(a: ExperimentResult, b: ExperimentResult) -> None:
    assert type(a) is type(b) and a.fitted == b.fitted and a.model == b.model and a.converged == b.converged
    assert np.array_equal(a.data, b.data) and a.subject == b.subject and a.requested == b.requested


# ---- 2.2: the machine equals the device with the drive keyword ---------------------------------------------------------------

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
def test_an_experiment_on_the_machine_equals_the_device_with_the_drive_keyword(
    name: str, fx: CircuitFixture
) -> None:
    on_machine = EXPERIMENTS[name](Machine(fx.device))
    with pytest.warns(
        QutipTrapDeprecationWarning, match=f"'gate_drive' argument of qutip_trap.experiments.*{name}"
    ):
        on_device = EXPERIMENTS[name](fx.device, gate_drive=fx.gate_drives[0])
    same_result(on_machine, on_device)
    assert isinstance(on_machine, KINDS[name]) and RESULT_TYPES[name] is KINDS[name]
    # every typed attribute equals value(key) where the fit produced the key
    for key in on_machine.fitted:
        attr = key if key.isidentifier() else None
        if attr is not None and isinstance(getattr(type(on_machine), attr, None), property):
            assert getattr(on_machine, attr) == on_machine.value(key), key
    assert on_machine.quality in ("good", "poor", "exact", "failed") and on_machine.created_at


def test_the_result_table_names_every_experiment_of_the_calibration_order() -> None:
    from qutip_trap.calibration import ALIASES, ORDER

    assert set(RESULT_TYPES) >= set(ORDER) | {"ramsey", "thermometry", "mode_spectroscopy", "ms_phase_scan"}
    assert {a for a in ALIASES if a in RESULT_TYPES} == {"thermometry", "mode_spectroscopy", "ms_phase_scan"}
    for cls in RESULT_TYPES.values():
        assert issubclass(cls, ExperimentResult) and cls.x_label and cls.y_label


def test_the_machine_supplies_table_options_and_builder_as_defaults(machine: Machine) -> None:
    device, kw = laboratory_kwargs(machine, {"shots": 5}, caller=rabi_scan)
    assert device is machine.device and kw["shots"] == 5
    assert kw["table"] is machine.table and kw["options"] == machine.numerics.to_solver_options(
        machine.physics
    )
    assert "builder_options" not in kw  # the machine's builder is None: nothing to supply
    _device, given = laboratory_kwargs(
        machine, {"table": None, "options": SolverOptions(atol=1e-12)}, caller=rabi_scan
    )
    assert given["table"] is None and given["options"].atol == 1e-12  # the call's own keywords win
    bare, kw2 = laboratory_kwargs(machine.device, {}, caller=rabi_scan)
    assert bare is machine.device and kw2 == {}
    assert as_machine(machine) is machine and as_machine(machine.device) == Machine(machine.device)
    with pytest.warns(QutipTrapDeprecationWarning, match="'gate_drives' argument"):
        laboratory_kwargs(machine, {"gate_drives": {}}, caller=rabi_scan)
    assert set(DRIVE_KEYWORDS) == {"gate_drive", "gate_drives", "entangling_drives"}


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


def test_calibrate_returns_the_report_on_a_machine_and_the_table_on_a_device(
    fx: CircuitFixture, machine: Machine
) -> None:
    scans: dict[str, Any] = {"pairs": [(0, 1)], "detection_records": 300, "detection_windows_s": WINDOWS}
    report = calibrate(Machine(fx.device, numerics=FAST), **scans)
    assert (
        isinstance(report, CalibrationReport) and report.table == machine.table and report.experiments == ()
    )
    assert report.results == {} and report.refused == {} and report.surrogate.table == report.table
    assert report.sample.sample_id == 0
    assert all(v == 0.0 for v in report.surrogate_error().values())  # the surrogate against itself
    table = calibrate(fx.device, **scans)
    assert isinstance(table, CalibrationTable) and table == machine.table
    with pytest.warns(
        QutipTrapDeprecationWarning, match="'surrogate' argument of qutip_trap.calibration.calibrate"
    ):
        assert calibrate(fx.device, surrogate=True, **scans) == table
    with pytest.warns(QutipTrapDeprecationWarning, match="calibrate_with_report is deprecated"):
        legacy = calibrate_with_report(fx.device, **scans)
    assert legacy.table == table
    with pytest.raises(ValueError, match="closed_form"):
        calibrate(machine, method="guess")  # type: ignore[arg-type]
    # the machine's t0_s is the calibration's default time
    late = dataclasses.replace(machine, physics=dataclasses.replace(machine.physics, t0_s=3.0))
    assert calibrate(late, **scans).table.fitted_at_s == pytest.approx(3.0)


def test_compile_with_report_is_the_deprecated_name_of_machine_compile(machine: Machine) -> None:
    with pytest.warns(QutipTrapDeprecationWarning, match="Machine\\(device\\).compile"):
        old = compile_with_report(BELL, machine.device)
    new = machine.compile(BELL)
    assert old.circuit == new.circuit == compile_report(BELL, machine.device).circuit
    assert old.n_entangling == new.n_entangling == 1


# ---- 2.2: the benchmarks on machines -------------------------------------------------------------------------------------------------


def test_benchmarks_take_a_machine_and_rewrite_the_run_kwargs(fx: CircuitFixture, machine: Machine) -> None:
    kind = kind_of("gpi2", (0,))
    ch = gate_channel(machine, kind)
    assert ch.kind == kind and ch.level == "GATE_LOCAL" and gate_channel(machine, kind) is ch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = gate_channel(
            fx.device, kind, table=machine.table, options=SolverOptions(branch_weight_min=1e-3)
        )
    texts = [str(w.message) for w in caught if issubclass(w.category, QutipTrapDeprecationWarning)]
    assert len(texts) == 2 and all("qutip_trap.benchmarks.budget.gate_channel" in t for t in texts)
    assert legacy is ch  # the same machine hash, the same cached channel
    rb = randomized_benchmarking(machine, (0,), (1, 2), n_sequences=1, shots=20, budget=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = randomized_benchmarking(
            fx.device,
            (0,),
            (1, 2),
            n_sequences=1,
            shots=20,
            budget=False,
            table=machine.table,
            options=SolverOptions(branch_weight_min=1e-3),
        )
    ours = sorted(
        str(w.message).split("'")[1] for w in caught if issubclass(w.category, QutipTrapDeprecationWarning)
    )
    assert ours == ["options", "table"]
    assert np.array_equal(rb.survival, old.survival) and rb.error_per_clifford == old.error_per_clifford
    with pytest.raises(TypeError, match="unexpected keyword"):
        randomized_benchmarking(machine, (0,), (1,), n_sequences=1, shots=5, budget=False, shotz=1)
    with pytest.raises(ValueError, match="pair=True"):
        randomized_benchmarking(machine, (0,), (1,), n_sequences=1, shots=5, budget=False, pair=True)
