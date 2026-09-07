"""Appendix E frozen as the public API (PLAN.md M0): every class, field, method and function it declares exists
in ``qutip_trap.api`` with the same names, every dataclass is frozen, and the prose amendments are in place."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re

import numpy as np
import pytest

from qutip_trap import api
from qutip_trap.provenance import repository_root

PLAN = repository_root() / "PLAN.md"

# fields the Appendix E prose adds to the code blocks (the "Run 5 additions" bullet list and the preamble)
PROSE_AMENDMENTS: dict[str, set[str]] = {
    "Level": {"lifetime_systematics_s"},
    "Transition": {"quadrupole_element_au", "quadrupole_convention"},
    "Beam": {"polarization_amplitudes", "modulation"},
    "Tone": {"theta_bessel_rad"},
    "Device": {"zones"},
    "Trap": {"dc_schedule", "basis_potentials"},
    "Schedule": {"transports"},
}

# declared in Appendix E as a property; implemented as a method taking the device's beams (documented deviation)
PROPERTY_AS_METHOD: dict[str, set[str]] = {"Drive": {"delta_k"}}


@dataclasses.dataclass
class DeclaredClass:
    fields: set[str] = dataclasses.field(default_factory=set)
    methods: set[str] = dataclasses.field(default_factory=set)
    frozen_dataclass: bool = False
    protocol: bool = False


def appendix_e_code_blocks() -> list[str]:
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("## Appendix E")
    section = text[start:]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def declared() -> tuple[dict[str, DeclaredClass], dict[str, list[str]]]:
    classes: dict[str, DeclaredClass] = {}
    functions: dict[str, list[str]] = {}
    for block in appendix_e_code_blocks():
        tree = ast.parse(block)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                dc = classes.setdefault(node.name, DeclaredClass())
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and getattr(deco.func, "id", "") == "dataclass":
                        dc.frozen_dataclass = any(
                            kw.arg == "frozen"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in deco.keywords
                        )
                if any(getattr(b, "id", "") == "Protocol" for b in node.bases):
                    dc.protocol = True
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        dc.fields.add(item.target.id)
                    elif isinstance(item, ast.FunctionDef):
                        dc.methods.add(item.name)
            elif isinstance(node, ast.FunctionDef):
                params = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                functions[node.name] = params
    return classes, functions


CLASSES, FUNCTIONS = declared()


def test_appendix_e_was_found_and_parsed() -> None:
    assert len(CLASSES) >= 50, sorted(CLASSES)
    assert len(FUNCTIONS) >= 20, sorted(FUNCTIONS)


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_declared_class_exists_with_its_fields_and_methods(name: str) -> None:
    decl = CLASSES[name]
    assert hasattr(api, name), f"Appendix E class {name} is missing from qutip_trap.api"
    cls = getattr(api, name)
    if decl.frozen_dataclass:
        assert dataclasses.is_dataclass(cls), f"{name} must be a dataclass"
        assert cls.__dataclass_params__.frozen, (
            f"{name} must be frozen (Appendix E: immutable after construction)"
        )
        have = {f.name for f in dataclasses.fields(cls)}
        missing = decl.fields - have
        assert not missing, f"{name} lacks Appendix E fields {sorted(missing)}"
        amend = PROSE_AMENDMENTS.get(name, set())
        assert amend <= have, f"{name} lacks the prose amendments {sorted(amend - have)}"
    for meth in decl.methods:
        assert hasattr(cls, meth), f"{name}.{meth} declared in Appendix E is missing"


def test_property_declared_as_method_is_documented() -> None:
    for cls_name, names in PROPERTY_AS_METHOD.items():
        cls = getattr(api, cls_name)
        for n in names:
            assert callable(getattr(cls, n))
            assert "Appendix E declares this as a property" in (getattr(cls, n).__doc__ or "")


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_declared_function_exists_with_its_parameters(name: str) -> None:
    assert hasattr(api, name), f"Appendix E function {name} is missing from qutip_trap.api"
    fn = getattr(api, name)
    have = set(inspect.signature(fn).parameters)
    want = set(FUNCTIONS[name])
    missing = want - have
    assert not missing, f"{name} lacks parameters {sorted(missing)}"


def test_every_dataclass_in_the_api_is_frozen() -> None:
    for name in api.__all__:
        obj = getattr(api, name)
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
            assert obj.__dataclass_params__.frozen, f"{name} is not frozen"


def test_instances_are_immutable() -> None:
    opts = api.SolverOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.atol = 1.0  # type: ignore[misc]


def test_unimplemented_entry_points_name_their_milestone() -> None:
    dev = object()
    # M6: the surrogate calibration, the compiler and the OpenQASM 2 importer are implemented; M8 the full simulated-experiment
    # calibration and Device.derived(); M9a the GATE_LOCAL tomography; the matrix-free kernel (M9b) still names its milestone
    with pytest.raises(ValueError, match=r"unknown calibration experiments"):
        api.calibrate(dev, surrogate=False, experiments=("not_an_experiment",))  # type: ignore[arg-type]
    assert api.load_openqasm2("OPENQASM 2.0; qreg q[1]; x q[0];").ops[0].name == "x"
    assert (
        api.compile_to_native(api.Circuit(1, (api.Operation("x", (0,), ()),), (0,)), dev).ops[0].name == "gpi"
    )  # type: ignore[arg-type]
    space = api.HilbertSpace((2,), (api.ModeTruncation(0, 4, (0, 1), 0.1),), None, ())
    assert len(space.operators().sigma_plus) == 1
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.dynamics.kernels import apply_drive_kernel

    assert "Section 5.4" in (JointExactEngine.process_tomography.__doc__ or "")
    with pytest.raises(ValueError, match=r"at least one pulse"):
        JointExactEngine().process_tomography(dev, (), space, None, None, api.SeedSpec(0))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match=r"milestone M9b"):
        apply_drive_kernel()
    from tests.fixtures import make_device, make_noise

    # the noise sampler of M7 is implemented: a quiet model returns a quiet sample
    assert make_noise().sample(np.random.default_rng(0), t_s=0.0, duration_s=1e-3, sample_id=0).is_quiet
    # M8: Device.derived() gathers the derived quantities with their ledger ids (the calibration's seeds)
    derived = make_device().derived()
    assert (
        derived.values["qubit_freq_hz[0]"] > 1e9
        and derived.provenance["qubit_freq_hz[0]"] == "conv.frequencies"
    )
    assert set(derived.values) == set(derived.provenance)
    # the atomic layer of M0a is implemented: this no longer raises
    assert api.species_by_name("171Yb+").zeeman_spectrum("S1/2", 5.0).labels
