"""Appendix E frozen as the public API (PLAN.md M0): every class, field, method and function it declares exists
in ``qutip_trap.api`` with the same names, every dataclass is frozen, and the prose amendments are in place."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
import sys
import textwrap

import numpy as np
import pytest

from qutip_trap import api
from qutip_trap.provenance import repository_root

PLAN = repository_root() / "PLAN.md"

RUNG_MODULES = (
    "qutip_trap.api",
    "qutip_trap",
    "qutip_trap.circuit",
    "qutip_trap.schedule",
    "qutip_trap.dynamics",
    "qutip_trap.physics",
    "qutip_trap.presets",
    "qutip_trap.io.qasm2",
    "qutip_trap.io.ionq",
    "qutip_trap.experiments",
    "qutip_trap.calibration",
    "qutip_trap.benchmarks",
)
"""Where a declared name is looked up: the Appendix E surface first, then the rung modules of the 2026-09 additions (which
``qutip_trap.api`` does not export; docs/api_implementation_plan.md 1.10)."""


def resolve(name: str) -> object:
    """The implementation of an Appendix E name: in ``qutip_trap.api``, else in the first rung module that exports it."""
    for module_name in RUNG_MODULES:
        module = importlib.import_module(module_name)
        if name in getattr(module, "__all__", ()):
            return getattr(module, name)
    raise AssertionError(f"Appendix E declares {name}, which no rung module exports")


# fields the Appendix E prose adds to the code blocks (the "Run 5 additions" bullet list and the preamble)
PROSE_AMENDMENTS: dict[str, set[str]] = {
    "Level": {"lifetime_systematics_s"},
    "Transition": {"quadrupole_element_au", "quadrupole_convention"},
    "Beam": {"polarization_amplitudes", "modulation"},
    "Tone": {"theta_bessel_rad"},
    "Device": {"zones", "roles"},
    "Trap": {"dc_schedule", "basis_potentials"},
    "Schedule": {"transports"},
}

# declared in Appendix E as a property; implemented as a method taking the device's beams (documented deviation)
PROPERTY_AS_METHOD: dict[str, set[str]] = {"Drive": {"delta_k"}}
# fields the Appendix E prose of the 2026-09 additions declares with a None default that reads as the declared type after
# construction (Circuit.measure, Circuit.registers): the default is None in the declaration and in the implementation
NONE_MEANS_DEFAULT: set[tuple[str, str]] = {("Circuit", "measure"), ("Circuit", "registers")}


# Appendix E declares these Literal types inline; the implementation names them (a type alias resolves to the same
# Literal, checked by resolving the alias in the defining module, so the comparison is by members, not by spelling)
# ---- deliberate divergences from Appendix E, each with the milestone that needed it (a widening never narrows a field) ----
ALLOWED_TYPE_DIVERGENCES: dict[tuple[str, str], tuple[str, str]] = {
    ("Drive", "kind"): (
        "Literal['gradient','light_shift','microwave','optical_E1','optical_E2','raman']",
        "M4: the Section 4.4.4 light-shift force is a drive kind of its own",
    ),
    ("Diagnostics", "mode_class"): (
        "dict[int,Literal['dropped','enr','frozen','resolved']]",
        "M9a: the ENR group is a fourth mode class (Section 5.1)",
    ),
    ("Tone", "envelope_hz"): (
        "Callable[[float],float]|ndarray|float",
        "M2: a square pulse is a constant envelope, not a one-point array",
    ),
    ("CollapseOp", "op"): ("Qobj|QobjEvo", "M7: a collapse operator with a time-dependent rate"),
    ("CombSpec", "pair_order_max"): (
        "int|None",
        "M2: None derives the sum depth from the envelope (ledger conv.comb_sum_depth); an int is the explicit override",
    ),
    ("CombSpec", "tau_convention"): (
        "Literal['field_fwhm','field_sech','intensity_fwhm','intensity_sech','intensity_sech2']",
        "M2: the two FWHM conventions the pulse-duration conversion accepts beside the three sech ones (Section 4.3.7)",
    ),
    ("Trap", "basis_potentials"): (
        "dict[str,Callable[...,float]]|None",
        "M0: the callable's signature is stated (a narrowing of the annotation, not of the value set)",
    ),
    ("Result", "bit_order"): (
        "Literal['qubit0_lsb','qubit0_msb']",
        "0.2.0: Result.reversed_bits() returns the same shots with qubit 0 leftmost for the SDKs that report that way "
        "(docs/api_implementation_plan.md 1.7); every run still reports qubit0_lsb",
    ),
}
# fields whose Appendix E default is deliberately not the implemented one, with the reason (ledger conv.appendix_e_signatures)
ALLOWED_DEFAULT_DIVERGENCES: dict[tuple[str, str], tuple[object, str]] = {
    ("CombSpec", "pair_order_max"): (
        None,
        "M2: Appendix E's literal 1200 is 2.6x too shallow at the plan's 80 MHz / 10 ps operating point; None derives "
        "the depth from the envelope (ledger conv.comb_sum_depth)",
    ),
}
# methods whose implementation takes a parameter Appendix E omits, with the reason (ledger conv.appendix_e_signatures)
EXTRA_REQUIRED_PARAMETERS: dict[tuple[str, str], dict[str, str]] = {
    ("MetastableChannels", "bbr_rate_hz"): {
        "species": "the channels object is species-agnostic; the line's Einstein coefficient, frequency and degeneracies "
        "live on the Species record (Section 4.5.7)"
    },
    ("Drive", "delta_k"): {
        "beams": "a Drive stores beam indices; the wavevectors live on Device.beams (Section 9.17)"
    },
}
# Appendix E names owned by milestone M12 (transport), whose methods legitimately raise NotImplementedError
M12_CLASSES = {"Zone", "VoltageWaveform", "FilterStage", "Transport"}
M12_METHODS = {("Trap", "pseudopotential_v"), ("Trap", "split_coefficients")}
# methods of the 2026-09 additions that name the later phase of docs/api_implementation_plan.md implementing them
LATER_PHASE_METHODS = {("Machine", "submit")}


@dataclasses.dataclass
class DeclaredField:
    annotation: str
    default: str | None  # the source text of the default, None when the field is required


@dataclasses.dataclass
class DeclaredClass:
    fields: dict[str, DeclaredField] = dataclasses.field(default_factory=dict)
    methods: dict[str, list[str]] = dataclasses.field(
        default_factory=dict
    )  # name -> declared parameters (no self)
    frozen_dataclass: bool = False
    protocol: bool = False


@dataclasses.dataclass
class DeclaredFunction:
    params: list[str]
    defaults: dict[str, str]  # parameter -> source text of its default


def appendix_e_code_blocks() -> list[str]:
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("## Appendix E")
    section = text[start:]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def _params(node: ast.FunctionDef, *, drop_self: bool) -> tuple[list[str], dict[str, str]]:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    names = [a.arg for a in positional] + [a.arg for a in args.kwonlyargs]
    if drop_self and names and names[0] == "self":
        names = names[1:]
    defaults: dict[str, str] = {}
    for a, d in zip(positional[len(positional) - len(args.defaults) :], args.defaults):
        defaults[a.arg] = ast.unparse(d)
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if d is not None:
            defaults[a.arg] = ast.unparse(d)
    return names, defaults


def declared() -> tuple[dict[str, DeclaredClass], dict[str, DeclaredFunction]]:
    classes: dict[str, DeclaredClass] = {}
    functions: dict[str, DeclaredFunction] = {}
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
                        dc.fields[item.target.id] = DeclaredField(
                            ast.unparse(item.annotation),
                            ast.unparse(item.value) if item.value is not None else None,
                        )
                    elif isinstance(item, ast.FunctionDef):
                        dc.methods[item.name] = _params(item, drop_self=True)[0]
            elif isinstance(node, ast.FunctionDef):
                params, defaults = _params(node, drop_self=False)
                functions[node.name] = DeclaredFunction(params, defaults)
    return classes, functions


CLASSES, FUNCTIONS = declared()

_LITERAL = re.compile(r"Literal\[([^\]]*)\]")


def _sorted_literals(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        members = sorted(x.strip() for x in m.group(1).split(","))
        return "Literal[" + ",".join(members) + "]"

    return _LITERAL.sub(repl, text)


def normalize_annotation(text: str, module: object | None = None) -> str:
    """One spelling for an annotation: no spaces or quotes, module prefixes dropped, type aliases of the implementing
    module resolved to their Literal, Literal members sorted."""
    s = text.replace('"', "").replace("'", "").replace(" ", "")
    for prefix in ("np.", "numpy.", "qt.", "qutip.", "typing."):
        s = s.replace(prefix, "")
    if module is not None:
        for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)):
            alias = getattr(module, name, None)
            if (
                alias is not None
                and getattr(alias, "__origin__", None) is not None
                and str(alias).startswith("typing.Literal")
            ):
                members = ",".join(repr(a) for a in alias.__args__)
                s = re.sub(rf"\b{name}\b", f"Literal[{members}]", s)
    s = s.replace('"', "").replace("'", "")
    return _sorted_literals(s)


def _default_matches(declared_src: str, f: dataclasses.Field[object]) -> bool:
    if declared_src.startswith("field(default_factory="):
        return f.default_factory is not dataclasses.MISSING  # type: ignore[comparison-overlap]
    try:
        value = eval(declared_src, {"__builtins__": {}}, {})  # noqa: S307  (Appendix E literals: numbers, strings, None, tuples)
    except Exception:  # noqa: BLE001
        return True  # a default the appendix writes symbolically is not comparable
    if f.default is dataclasses.MISSING:
        return False
    if isinstance(value, float) or isinstance(f.default, float):
        return float(value) == float(f.default)  # type: ignore[arg-type]
    return bool(value == f.default)


def test_appendix_e_was_found_and_parsed() -> None:
    assert len(CLASSES) >= 50, sorted(CLASSES)
    assert len(FUNCTIONS) >= 20, sorted(FUNCTIONS)


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_declared_class_exists_with_its_fields_types_defaults_and_methods(name: str) -> None:
    decl = CLASSES[name]
    cls = resolve(name)
    assert inspect.isclass(cls), f"Appendix E declares the class {name}; the surface has {cls!r}"
    module = sys.modules[cls.__module__]
    if decl.frozen_dataclass:
        assert dataclasses.is_dataclass(cls), f"{name} must be a dataclass"
        assert cls.__dataclass_params__.frozen, (
            f"{name} must be frozen (Appendix E: immutable after construction)"
        )
        have = {f.name: f for f in dataclasses.fields(cls)}
        missing = set(decl.fields) - set(have)
        assert not missing, f"{name} lacks Appendix E fields {sorted(missing)}"
        amend = PROSE_AMENDMENTS.get(name, set())
        assert amend <= set(have), f"{name} lacks the prose amendments {sorted(amend - set(have))}"
        for fname, dfield in decl.fields.items():
            impl = have[fname]
            want = normalize_annotation(dfield.annotation)
            got = normalize_annotation(str(impl.type), module)
            if (name, fname) in ALLOWED_TYPE_DIVERGENCES:
                allowed, _reason = ALLOWED_TYPE_DIVERGENCES[(name, fname)]
                assert got == normalize_annotation(allowed), (
                    f"{name}.{fname}: implemented as {got!r}, the recorded divergence is {allowed!r}"
                )
            else:
                assert got == want, f"{name}.{fname}: Appendix E declares {want!r}, implemented as {got!r}"
            if dfield.default is None:
                # a required field in Appendix E may gain a default (additive), never the other way round
                continue
            assert (
                impl.default is not dataclasses.MISSING or impl.default_factory is not dataclasses.MISSING
            ), (  # type: ignore[comparison-overlap]
                f"{name}.{fname} has the default {dfield.default} in Appendix E but is required here"
            )
            if (name, fname) in ALLOWED_DEFAULT_DIVERGENCES:
                allowed_default, _why = ALLOWED_DEFAULT_DIVERGENCES[(name, fname)]
                assert impl.default == allowed_default, (
                    f"{name}.{fname}: implemented default {impl.default!r}, the recorded divergence is {allowed_default!r}"
                )
                continue
            if (name, fname) in NONE_MEANS_DEFAULT:
                assert impl.default is None, f"{name}.{fname}: the omitted argument arrives as None"
                continue
            assert _default_matches(dfield.default, impl), (
                f"{name}.{fname}: Appendix E default {dfield.default}, implemented default {impl.default!r}"
            )
        # additive fields (not in Appendix E) must carry a default, so an Appendix E constructor call still works
        for fname, impl in have.items():
            if fname not in decl.fields and fname not in PROSE_AMENDMENTS.get(name, set()):
                assert (
                    impl.default is not dataclasses.MISSING or impl.default_factory is not dataclasses.MISSING
                ), (  # type: ignore[comparison-overlap]
                    f"{name}.{fname} is not in Appendix E and has no default: an Appendix E construction would fail"
                )
    for meth, declared_params in decl.methods.items():
        assert hasattr(cls, meth), f"{name}.{meth} declared in Appendix E is missing"
        attr = inspect.getattr_static(cls, meth)
        if isinstance(attr, property) or name in PROPERTY_AS_METHOD and meth in PROPERTY_AS_METHOD[name]:
            continue
        if decl.protocol:
            continue  # the Protocol's implementations are checked where they are used
        sig = inspect.signature(getattr(cls, meth))
        params = [p for p in sig.parameters.values() if p.name != "self"]
        names_ = {p.name for p in params}
        missing = set(declared_params) - names_
        assert not missing, f"{name}.{meth} lacks the Appendix E parameters {sorted(missing)}"
        extra_required = [
            p.name
            for p in params
            if p.name not in declared_params
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        allowed = EXTRA_REQUIRED_PARAMETERS.get((name, meth), {})
        assert set(extra_required) <= set(allowed), (
            f"{name}.{meth} requires {extra_required} which Appendix E does not declare (record it in "
            "EXTRA_REQUIRED_PARAMETERS with the reason, or give it a default)"
        )


def _is_stub(func: object) -> str | None:
    """The message of a `raise NotImplementedError(...)` that is the whole body of ``func``, else None."""
    try:
        src = inspect.getsource(func)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return None
    tree = ast.parse(textwrap.dedent(src))
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        return None
    body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].exc is not None:
        exc = body[0].exc
        if isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "NotImplementedError":
            return ast.unparse(exc)
    return None


def test_no_appendix_e_method_outside_m12_is_a_stub() -> None:
    """Every Appendix E method that is not owned by milestone M12 (transport, Section 4.6) has a body: a method that raises
    NotImplementedError as its whole body is an unfinished M0-M10 deliverable (the 'freeze' is of working interfaces)."""
    stubs: list[str] = []
    for name, decl in CLASSES.items():
        if name in M12_CLASSES or decl.protocol:
            continue
        cls = resolve(name)
        for meth in decl.methods:
            if (name, meth) in LATER_PHASE_METHODS:
                continue
            if (name, meth) in M12_METHODS:
                continue
            attr = inspect.getattr_static(cls, meth)
            func = attr.fget if isinstance(attr, property) else attr
            msg = _is_stub(func)
            if msg is not None:
                stubs.append(f"{name}.{meth}: {msg}")
    assert not stubs, "Appendix E methods that are still stubs:\n  " + "\n  ".join(stubs)


def test_property_declared_as_method_is_documented() -> None:
    for cls_name, names in PROPERTY_AS_METHOD.items():
        cls = resolve(cls_name)
        for n in names:
            assert callable(getattr(cls, n))
            assert "Appendix E declares this as a property" in (getattr(cls, n).__doc__ or "")


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_declared_function_exists_with_its_parameters_and_defaults(name: str) -> None:
    fn = resolve(name)
    decl = FUNCTIONS[name]
    sig = inspect.signature(fn)
    have = sig.parameters
    missing = set(decl.params) - set(have)
    assert not missing, f"{name} lacks parameters {sorted(missing)}"
    extra_required = [
        p.name
        for p in have.values()
        if p.name not in decl.params
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not extra_required, f"{name} requires {extra_required} which Appendix E does not declare"
    for pname, dsrc in decl.defaults.items():
        try:
            want = eval(dsrc, {"__builtins__": {}}, {})  # noqa: S307
        except Exception:  # noqa: BLE001
            continue
        got = have[pname].default
        assert got is not inspect.Parameter.empty, (
            f"{name}({pname}) has the default {dsrc} in Appendix E, none here"
        )
        if isinstance(want, tuple) and isinstance(got, tuple):
            assert set(want) <= set(got) or want == got, (
                f"{name}({pname}): {dsrc} declared, {got!r} implemented"
            )
        else:
            assert got == want, f"{name}({pname}): Appendix E default {dsrc}, implemented {got!r}"


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
    # calibration and Device.derived(); M9a the GATE_LOCAL tomography; M9b the matrix-free kernel
    with pytest.raises(ValueError, match=r"unknown calibration experiments"):
        api.calibrate(dev, method="experiments", experiments=("not_an_experiment",))  # type: ignore[arg-type]
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
    # M9b: the kernel applies a factorized drive operator mode by mode (Section 11.3 item 4)
    fact = space.drive_operator_factorized(0, {0: 0.05})
    ket = space.initial_state([0], fock={0: 1}).joint
    assert ket is not None
    assert (apply_drive_kernel(fact, ket) - space.drive_operator(0, {0: 0.05}) * ket).norm() < 1e-13
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
