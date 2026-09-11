"""The public surface, name by name (docs/api_implementation_plan.md item 0.2, after Cirq's ``json_test_data/spec.py``, which
forces a decision per name): every ``__all__`` name of every rung module resolves, carries a docstring of its own and appears in
backticks on a documentation page, and every frozen dataclass among them round-trips through ``dataclasses.replace``.

Phase 1 adds the rung modules to ``RUNG_MODULES`` as it creates them. A name of the 0.1.0 surface that has no line under
``docs/`` yet is listed in ``NOT_YET_DOCUMENTED``; the list may only shrink (a name leaves it when its line is written, and a
new public name is never added to it), and Phase 3.4 empties it when every rung name gets its page."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from qutip_trap import api
from qutip_trap.provenance import repository_root
from tests.fixtures import (
    make_calibration_table,
    make_crystal,
    make_detector,
    make_device,
    make_diagnostics,
    make_field,
    make_hardware,
    make_noise,
    make_raman_pair,
    make_result,
    make_run_state,
    make_space,
    make_trap,
)

ROOT = repository_root()
DOCS = ROOT / "docs"

RUNG_MODULES: tuple[str, ...] = ("qutip_trap.api",)
"""The modules whose ``__all__`` is the public surface. 1.6 adds ``qutip_trap``, ``qutip_trap.circuit``, ``.schedule``,
``.dynamics``, ``.physics``, ``.presets``, ``.io.qasm2`` and ``.io.ionq``; 3.3 adds ``qutip_trap.experimental``."""

PAGES: tuple[Path, ...] = tuple(
    DOCS / name for name in ("physics_notes.md", "conventions.md", "examples.md", "limits.md", "README.md")
)
"""The documentation pages a name must appear on (the two planning documents under docs/ are not pages)."""

NOT_YET_DOCUMENTED: dict[str, frozenset[str]] = {
    "qutip_trap.api": frozenset(
        {
            "AdaptiveML",
            "AnharmonicTerms",
            "ApparatusPreset",
            "AtomicStructure",
            "Beam",
            "BenchmarkBudget",
            "BlochModel",
            "BudgetLine",
            "CachedOperators",
            "CalEntry",
            "CalibrationCache",
            "CalibrationError",
            "CalibrationReport",
            "CalibrationRun",
            "CalibrationScans",
            "CalibrationTable",
            "CameraGeometry",
            "ChannelSummary",
            "ClockPoint",
            "CollapseOp",
            "CollisionEvent",
            "Collisions",
            "CombSpec",
            "CompileReport",
            "CompositePulse",
            "ControlSegment",
            "DarkStateReport",
            "DcElectrodes",
            "DecouplingSequence",
            "DerivedQuantities",
            "DetectionCalibration",
            "DetectionRates",
            "Detector",
            "DevicePreset",
            "Drift",
            "Drive",
            "Electrodes",
            "ExperimentResult",
            "Field",
            "FilterStage",
            "FirstPhoton",
            "FluorescenceRates",
            "GHZResult",
            "GateChannel",
            "GateCheck",
            "GateDrive",
            "GateLocalReport",
            "GateModes",
            "GateStep",
            "GateTarget",
            "Gauss",
            "HardwareChain",
            "HilbertSpace",
            "IncompleteSpeciesTable",
            "InternalLevels",
            "KAK",
            "LightShiftCouplings",
            "Mains",
            "MathieuParameters",
            "MetastableChannels",
            "MicromotionIndex",
            "Mode",
            "ModeSpec",
            "ModeTruncation",
            "MotionalModel",
            "MultiLevelOptions",
            "NoiseModel",
            "NoiseSample",
            "NoiseSpectrum",
            "Observation",
            "POVM",
            "PhotonRecord",
            "PlayedGate",
            "PolGradientBeams",
            "PolarizationModulation",
            "PreparationRecipe",
            "PreparationRun",
            "Pulse",
            "PulseEngine",
            "QVCircuit",
            "RBResult",
            "RBSequence",
            "ReadoutBudget",
            "ReadoutErrors",
            "ReadoutOutcome",
            "ReadoutScheme",
            "RecordModel",
            "RfDrive",
            "RunRecord",
            "RunState",
            "ScatteringOptions",
            "ScheduledEvent",
            "SeedSpec",
            "Segment",
            "ShapedPulse",
            "SidebandCoolingSpec",
            "SpaceSelection",
            "Species",
            "State",
            "SteadyStateReport",
            "StepChannel",
            "SurrogateReport",
            "Tesla",
            "ThresholdDiscriminator",
            "TimeResolvedML",
            "Tone",
            "Traces",
            "Trajectory",
            "TransportBudget",
            "Trap",
            "TwoQubitClifford",
            "Waveform",
            "ZeemanSpectrum",
            "ZigzagError",
            "apply_hardware_chain",
            "available",
            "ca40_optical_recipe",
            "calibrate_detection",
            "calibrate_entangling_angle",
            "calibrate_with_report",
            "choi_from_unitary",
            "choi_least_squares",
            "circuit_unitary",
            "clear_budget_cache",
            "collision_rate_per_ion",
            "composite_pulse",
            "crosstalk_scan",
            "crystal_image",
            "decompose_two_qubit_clifford",
            "decompose_two_qubit_unitary",
            "decoupling_sequence",
            "depolarizing_choi",
            "depolarizing_rate",
            "derive_light_shift_drive",
            "derive_raman_drive",
            "design_waveform",
            "detection_histogram",
            "detection_rates_for_ion",
            "entanglement_infidelity",
            "exact_gate_check",
            "field_scan",
            "filter_function",
            "frame_rotated",
            "frozen_excitation_bounds",
            "full_calibration",
            "gate_modes",
            "gate_space",
            "gate_steps",
            "gaussian_spectrum",
            "ghz_circuit",
            "haar_random_unitary",
            "heating_rate",
            "input_states",
            "kraus_operators",
            "last_record",
            "micromotion_scan",
            "mode_spectroscopy",
            "ms_phase_scan",
            "ms_scan",
            "optimize_threshold",
            "ou_spectrum",
            "parity_circuit",
            "parity_scan",
            "pauli_twirl",
            "physical_schedule",
            "povm_for",
            "power_law_spectrum",
            "prepare",
            "project_cptp",
            "ramsey",
            "ramsey_frequency",
            "random_square_circuit",
            "random_two_qubit_clifford",
            "resolve_level",
            "run_preparation",
            "scattering_estimates",
            "scattering_rate",
            "select_space",
            "sideband_spectroscopy",
            "solve_amplitude_modulation",
            "solve_crystal",
            "solve_fourier_amplitude_modulation",
            "solve_frequency_modulation",
            "species_by_name",
            "split_feasible",
            "standard_recipe",
            "stark_scan",
            "step_space",
            "surrogate_table",
            "thermal_robustness",
            "thermometry",
            "transport_budget",
            "white_spectrum",
        }
    )
}
"""The names of the 0.1.0 surface with no backticked mention on a documentation page (2026-09-11: 193 of 237)."""


def _module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _surface(module_name: str) -> list[tuple[str, object]]:
    module = _module(module_name)
    return [(name, getattr(module, name)) for name in module.__all__]


@cache
def _source_modules(module_name: str) -> dict[str, str]:
    """Exported name -> the module that defines it, from the rung module's ``from ... import`` statements (a name the rung
    module defines itself maps to the rung module)."""
    module = _module(module_name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.ClassDef | ast.FunctionDef):
            targets = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                else [node.name]
            )
            for t in targets:
                out[t] = module_name
    return out


def _attribute_docstring(module_name: str, name: str) -> str | None:
    """The string literal that follows ``name = ...`` at the top level of the module (a type alias's or NewType's docstring)."""
    module = _module(module_name)
    assert module.__file__ is not None
    body = ast.parse(Path(module.__file__).read_text(encoding="utf-8")).body
    for node, following in zip(body, body[1:]):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if name in targets:
            if (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                return following.value.value
            return None
    return None


def _own_docstring(module_name: str, name: str, obj: object) -> str | None:
    """The docstring ``name`` itself carries: a class's own (never an inherited one, never the signature a dataclass
    generates), a function's, or the attribute docstring of a type alias or NewType in its defining module."""
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__")
        if not isinstance(doc, str):
            return None
        if dataclasses.is_dataclass(obj) and doc.startswith(obj.__name__ + "("):
            return None  # dataclasses.dataclass writes the signature when the class has no docstring
        return doc
    if inspect.isroutine(obj):
        doc = getattr(obj, "__doc__", None)
        return doc if isinstance(doc, str) else None
    return _attribute_docstring(_source_modules(module_name)[name], name)


@cache
def _code_spans() -> tuple[str, ...]:
    """Every backticked span of the documentation pages: fenced blocks and inline code."""
    spans: list[str] = []
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        spans += re.findall(r"```[A-Za-z]*\n(.*?)```", text, flags=re.S)
        spans += re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", text, flags=re.S))
    return tuple(spans)


def _documented(name: str) -> bool:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    return any(pattern.search(span) for span in _code_spans())


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_resolves(module_name: str) -> None:
    module = _module(module_name)
    assert len(module.__all__) == len(set(module.__all__)), "a name is exported twice"
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.__all__ names {name!r}, which does not resolve"


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_carries_a_docstring_of_its_own(module_name: str) -> None:
    missing = [
        name
        for name, obj in _surface(module_name)
        if not (_own_docstring(module_name, name, obj) or "").strip()
    ]
    assert not missing, f"public names of {module_name} without a docstring: {missing}"


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_is_documented_or_listed_as_not_yet(module_name: str) -> None:
    """A name is public when it is documented (Qiskit's rule): it appears in backticks, alone or inside a code span or a
    fenced block, on one of the documentation pages. The 0.1.0 names that still lack a line are listed above; this test
    fails both ways, on a name that lost its line and on a listed name that gained one (remove it from the list)."""
    undocumented = {name for name in _module(module_name).__all__ if not _documented(name)}
    listed = NOT_YET_DOCUMENTED.get(module_name, frozenset())
    new = sorted(undocumented - listed)
    assert not new, f"public names of {module_name} with no line under docs/ (write the line): {new}"
    stale = sorted(listed - undocumented)
    assert not stale, (
        f"names now documented, or no longer public: remove them from NOT_YET_DOCUMENTED: {stale}"
    )


def _dataclasses_of(module_name: str) -> list[tuple[str, type]]:
    return [
        (name, obj)
        for name, obj in _surface(module_name)
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj)
    ]


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_dataclass_is_frozen_and_rebuilt_from_its_own_fields(module_name: str) -> None:
    """``dataclasses.replace(obj)`` calls ``type(obj)(**{f.name: getattr(obj, f.name)})`` for the ``init`` fields: it
    reproduces ``obj`` when and only when the constructor takes exactly the fields (no ``init=False`` field, no ``InitVar``, no
    hand-written ``__init__``), which is what ``dataclasses.replace(machine, level=...)`` relies on from 0.2.0."""
    for name, cls in _dataclasses_of(module_name):
        assert cls.__dataclass_params__.frozen, f"{name} is not frozen"
        fields = dataclasses.fields(cls)
        assert all(f.init for f in fields), f"{name} has an init=False field"
        assert set(inspect.signature(cls).parameters) == {f.name for f in fields}, (
            f"{name}'s constructor does not take exactly its fields"
        )


def _equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _instances() -> list[object]:
    """Frozen dataclasses of the surface as the shared fixtures build them, plus the cheap ones built here."""
    table = make_calibration_table()
    return [
        make_device(),
        make_crystal(),
        make_trap(),
        make_field(),
        make_noise(),
        make_detector(),
        make_hardware(),
        *make_raman_pair(),
        make_run_state(),
        table,
        table.field,
        make_space(),
        make_diagnostics(),
        make_result(np.array([[0, 1], [1, 1], [0, 0]], dtype=np.uint8)),
        api.Circuit(2, (api.Operation("h", (0,), ()), api.Operation("cnot", (0, 1), ())), (0, 1)),
        api.SolverOptions(),
        api.ModeTruncation(0, 4, (0, 1), 0.1),
    ]


@pytest.mark.parametrize("instance", _instances(), ids=lambda x: type(x).__name__)
def test_fixture_instances_round_trip_through_replace(instance: object) -> None:
    copy = dataclasses.replace(instance)  # type: ignore[type-var]  (every instance is a dataclass)
    assert type(copy) is type(instance)
    assert _equal(copy, instance), type(instance).__name__
