"""``Device.to_dict`` and ``Device.from_dict``: the JSON form of the device record and its JSON schema
(docs/api_implementation_plan.md 2.5; docs/api_proposal.md Section 4.6, after Pulser's ``Device`` serialisation).

The device tree is dataclasses over simple leaves, so one walker driven by the field annotations (strings, under
``from __future__ import annotations``) writes and reads every record: a dataclass is an object of its fields, a tuple a
list, a ``dict`` an object whose keys are written as strings (an ``int`` key as its digits, a tuple key joined by commas),
a ``Literal`` its value, a float a JSON number (the non-finite values as the strings ``"inf"``, ``"-inf"``, ``"nan"``, which
JSON has no number for), a complex ``{"re", "im"}``, a numpy array ``{"dtype", "shape", "data"}`` with its values flattened
in C order; ``None`` is ``null``. A field that holds callables (``Trap.basis_potentials``, the M12 basis potentials) has no
JSON form and is refused when set. The same grammar generates ``docs/schemas/device.schema.json`` (``tools/schemas.py``),
so a published simulator configuration is citable and diffable outside Python; ``Device.hash()`` is the identity the
envelope carries, and a record read back has the hash it was written with.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from qutip_trap.device.model import Device

SCHEMA_VERSION = 1
"""The version of the device envelope: ``{"schema_version", "qutip_trap_version", "device_hash", "device"}``."""

_SCALARS = {"float", "int", "str", "bool", "complex"}
_LEAF_ALIASES = {"Family", "Kind", "LeakPolicy", "DriveKind", "WaveformKind", "Leg", "Entangler"}
"""Type aliases of ``Literal[...]`` strings the tree uses; read and written as strings."""
_NON_FINITE = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


# ---- the annotation grammar -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One parsed annotation: ``kind`` in {union, none, tuple, dict, literal, array, callable, scalar, cls, alias, any}."""

    kind: str
    args: tuple[Any, ...] = ()
    name: str = ""
    variadic: bool = False


def _split(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def parse(annotation: str) -> Node:
    """The ``Node`` of one annotation string of the device tree; an unknown form is refused, never guessed."""
    text = annotation.strip().strip('"').strip("'")
    if "|" in text and _split(text, "|") != [text]:
        return Node("union", tuple(parse(p) for p in _split(text, "|")))
    if text == "None":
        return Node("none")
    if text in ("np.ndarray", "ndarray", "numpy.ndarray"):
        return Node("array")
    if text.startswith("Callable"):
        return Node("callable")
    if text == "Any":
        return Node("any")
    if text in _SCALARS:
        return Node("scalar", name=text)
    if text.startswith("Literal[") and text.endswith("]"):
        values = tuple(v.strip().strip("'\"") for v in _split(text[len("Literal[") : -1], ","))
        return Node("literal", values)
    for head in ("tuple[", "Sequence["):
        if text.startswith(head) and text.endswith("]"):
            items = _split(text[len(head) : -1], ",")
            if head == "Sequence[":
                return Node("tuple", (parse(items[0]),), variadic=True)
            if len(items) == 2 and items[1] == "...":
                return Node("tuple", (parse(items[0]),), variadic=True)
            return Node("tuple", tuple(parse(i) for i in items))
    for head in ("dict[", "Mapping["):
        if text.startswith(head) and text.endswith("]"):
            k, v = _split(text[len(head) : -1], ",")
            return Node("dict", (parse(k), parse(v)))
    if text in _LEAF_ALIASES:
        return Node("alias", name=text)
    if text.isidentifier():
        return Node("cls", name=text)
    raise ValueError(f"Device.to_dict: no JSON form for the annotation {annotation!r}")


# ---- the dataclass registry ---------------------------------------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def _registry() -> dict[str, type]:
    """Every dataclass of the loaded package modules, by class name (the presets load every class a device holds)."""
    if _REGISTRY:
        return _REGISTRY
    import sys

    import qutip_trap.device.presets  # noqa: F401

    for modname, mod in list(sys.modules.items()):
        if modname.startswith("qutip_trap.") and mod is not None:
            for name, obj in vars(mod).items():
                if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__ == modname:
                    if _REGISTRY.setdefault(name, obj) is not obj:
                        raise TypeError(
                            f"two dataclasses named {name!r}: {_REGISTRY[name].__module__}, {modname}"
                        )
    return _REGISTRY


def resolve(name: str) -> type:
    reg = _registry()
    if name not in reg:
        raise ValueError(
            f"Device.to_dict: the dataclass {name!r} is not on the Appendix E surface or beside one that is"
        )
    return reg[name]


def _annotation(f: dataclasses.Field[Any]) -> str:
    return f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))


# ---- encoding -----------------------------------------------------------------------------------------------------------------------


def _float(value: float) -> float | str:
    x = float(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _key_out(key: Any) -> str:
    if isinstance(key, tuple):
        return ",".join(
            str(int(k)) if isinstance(k, (int, np.integer)) and not isinstance(k, bool) else str(k)
            for k in key
        )
    if isinstance(key, bool):
        return str(key).lower()
    return str(key)


def encode(value: Any, node: Node, where: str) -> Any:
    """The JSON form of ``value`` under ``node``; ``where`` names the field for the error messages."""
    if node.kind == "union":
        if value is None and any(n.kind == "none" for n in node.args):
            return None
        for n in node.args:
            if n.kind != "none" and _matches(value, n):
                return encode(value, n, where)
        raise ValueError(
            f"Device.to_dict: {where} holds {type(value).__name__}, not one of the declared types"
        )
    if node.kind == "none":
        if value is not None:
            raise ValueError(f"Device.to_dict: {where} should be None")
        return None
    if node.kind == "callable":
        if value is None:
            return None
        raise ValueError(
            f"Device.to_dict: {where} holds callables (the M12 basis potentials), which have no JSON form"
        )
    if node.kind == "any":
        return value
    if node.kind == "scalar":
        if node.name == "float":
            return _float(value)
        if node.name == "complex":
            if not isinstance(value, complex):
                return _float(
                    value
                )  # a real number in a complex-typed slot stays real: the digest tells them apart
            return {"re": _float(value.real), "im": _float(value.imag)}
        if node.name == "int":
            return int(value)
        if node.name == "bool":
            return bool(value)
        return str(value)
    if node.kind in ("literal", "alias"):
        return value
    if node.kind == "array":
        arr = np.asarray(value)
        flat = arr.reshape(-1)
        data = (
            [{"re": _float(complex(x).real), "im": _float(complex(x).imag)} for x in flat]
            if np.iscomplexobj(arr)
            else [_float(x) for x in flat]
        )
        return {"dtype": str(arr.dtype), "shape": [int(n) for n in arr.shape], "data": data}
    if node.kind == "tuple":
        items = list(value)
        if node.variadic:
            return [encode(v, node.args[0], f"{where}[{i}]") for i, v in enumerate(items)]
        if len(items) != len(node.args):
            raise ValueError(
                f"Device.to_dict: {where} has {len(items)} items, the annotation {len(node.args)}"
            )
        return [encode(v, n, f"{where}[{i}]") for i, (v, n) in enumerate(zip(items, node.args))]
    if node.kind == "dict":
        _k, v = node.args
        return {_key_out(key): encode(val, v, f"{where}[{key!r}]") for key, val in value.items()}
    if node.kind == "cls":
        cls = resolve(node.name)
        if not isinstance(value, cls):
            raise ValueError(f"Device.to_dict: {where} holds {type(value).__name__}, not {node.name}")
        record = cast("DataclassInstance", value)
        return {
            f.name: encode(getattr(record, f.name), parse(_annotation(f)), f"{where}.{f.name}")
            for f in dataclasses.fields(record)
        }
    raise ValueError(f"Device.to_dict: unhandled node {node.kind}")


def _matches(value: Any, node: Node) -> bool:
    if node.kind == "scalar":
        return {
            "float": isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool),
            "int": isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            "str": isinstance(value, str),
            "bool": isinstance(value, (bool, np.bool_)),
            "complex": isinstance(value, (complex, float, int, np.number)),
        }[node.name]
    if node.kind in ("literal", "alias"):
        return isinstance(value, str)
    if node.kind == "array":
        return isinstance(value, np.ndarray)
    if node.kind == "tuple":
        return isinstance(value, (tuple, list))
    if node.kind == "dict":
        return isinstance(value, Mapping)
    if node.kind == "cls":
        return isinstance(value, resolve(node.name))
    if node.kind == "callable":
        return callable(value)
    return node.kind == "any"


# ---- decoding -----------------------------------------------------------------------------------------------------------------------


def _unfloat(value: Any) -> float:
    if isinstance(value, str):
        return _NON_FINITE[value]
    return float(value)


def _key_in(text: str, node: Node) -> Any:
    if node.kind == "scalar" and node.name == "int":
        return int(text)
    if node.kind == "scalar" and node.name == "bool":
        return text == "true"
    if node.kind == "tuple":
        parts = text.split(",") if text else []
        if node.variadic:
            return tuple(_key_in(p, node.args[0]) for p in parts)
        return tuple(_key_in(p, n) for p, n in zip(parts, node.args))
    return text


def decode(data: Any, node: Node, where: str) -> Any:
    """The Python value of the JSON form ``data`` under ``node``."""
    if node.kind == "union":
        if data is None and any(n.kind == "none" for n in node.args):
            return None
        errors: list[str] = []
        for n in node.args:
            if n.kind == "none":
                continue
            if _json_matches(data, n):
                try:
                    return decode(data, n, where)
                except (ValueError, TypeError, KeyError) as exc:  # a later alternative may fit
                    errors.append(str(exc))
        raise ValueError(
            f"Device.from_dict: {where}: no declared type fits {type(data).__name__} ({'; '.join(errors)})"
        )
    if node.kind == "none":
        return None
    if node.kind == "callable":
        if data is None:
            return None
        raise ValueError(f"Device.from_dict: {where} holds callables, which have no JSON form")
    if node.kind == "any":
        return data
    if node.kind == "scalar":
        if node.name == "float":
            return _unfloat(data)
        if node.name == "complex":
            if not isinstance(data, dict):
                return _unfloat(data)
            return complex(_unfloat(data["re"]), _unfloat(data["im"]))
        if node.name == "int":
            return int(data)
        if node.name == "bool":
            return bool(data)
        return str(data)
    if node.kind in ("literal", "alias"):
        if node.kind == "literal" and data not in node.args:
            raise ValueError(f"Device.from_dict: {where} = {data!r} is not one of {node.args}")
        return data
    if node.kind == "array":
        dtype = np.dtype(data["dtype"])
        if dtype.kind == "c":
            values = [complex(_unfloat(x["re"]), _unfloat(x["im"])) for x in data["data"]]
        else:
            values = [_unfloat(x) for x in data["data"]]
        return np.asarray(values, dtype=dtype).reshape(tuple(int(n) for n in data["shape"]))
    if node.kind == "tuple":
        items = list(data)
        if node.variadic:
            return tuple(decode(v, node.args[0], f"{where}[{i}]") for i, v in enumerate(items))
        if len(items) != len(node.args):
            raise ValueError(
                f"Device.from_dict: {where} has {len(items)} items, the annotation {len(node.args)}"
            )
        return tuple(decode(v, n, f"{where}[{i}]") for i, (v, n) in enumerate(zip(items, node.args)))
    if node.kind == "dict":
        k, v = node.args
        return {_key_in(key, k): decode(val, v, f"{where}[{key}]") for key, val in data.items()}
    if node.kind == "cls":
        cls = resolve(node.name)
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"Device.from_dict: {where} ({node.name}) has unknown fields {unknown}")
        kwargs = {
            name: decode(data[name], parse(_annotation(f)), f"{where}.{name}")
            for name, f in fields.items()
            if name in data
        }
        return cls(**kwargs)
    raise ValueError(f"Device.from_dict: unhandled node {node.kind}")


def _json_matches(data: Any, node: Node) -> bool:
    if node.kind == "scalar":
        if node.name == "float":
            return isinstance(data, (int, float)) and not isinstance(data, bool) or data in _NON_FINITE
        if node.name == "int":
            return isinstance(data, int) and not isinstance(data, bool)
        if node.name == "bool":
            return isinstance(data, bool)
        if node.name == "complex":
            return (
                (isinstance(data, dict) and set(data) == {"re", "im"})
                or (isinstance(data, (int, float)) and not isinstance(data, bool))
                or data in _NON_FINITE
            )
        return isinstance(data, str)
    if node.kind in ("literal", "alias"):
        return isinstance(data, str) and (node.kind == "alias" or data in node.args)
    if node.kind == "array":
        return isinstance(data, dict) and set(data) == {"dtype", "shape", "data"}
    if node.kind == "tuple":
        return isinstance(data, list)
    if node.kind in ("dict", "cls"):
        return isinstance(data, dict)
    return node.kind == "any"


# ---- the envelope -------------------------------------------------------------------------------------------------------------------


def device_to_dict(device: Device) -> dict[str, Any]:
    from qutip_trap import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "qutip_trap_version": __version__,
        "device_hash": device.hash(),
        "device": encode(device, Node("cls", name="Device"), "device"),
    }


def device_from_dict(data: Mapping[str, Any]) -> Device:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Device.from_dict reads schema version {SCHEMA_VERSION}, got {version!r}")
    device: Device = decode(data["device"], Node("cls", name="Device"), "device")
    return device


# ---- the JSON schema ------------------------------------------------------------------------------------------------------------------

_FLOAT_SCHEMA: dict[str, Any] = {
    "anyOf": [{"type": "number"}, {"type": "string", "enum": ["inf", "-inf", "nan"]}]
}
_COMPLEX_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "required": ["re", "im"],
            "additionalProperties": False,
            "properties": {"re": _FLOAT_SCHEMA, "im": _FLOAT_SCHEMA},
        },
        _FLOAT_SCHEMA,
    ]
}
_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["dtype", "shape", "data"],
    "additionalProperties": False,
    "properties": {
        "dtype": {"type": "string"},
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "data": {"type": "array", "items": {"anyOf": [_FLOAT_SCHEMA, _COMPLEX_SCHEMA]}},
    },
}


def schema_of(node: Node, defs: dict[str, Any]) -> dict[str, Any]:
    """The JSON schema of one node; dataclasses become ``$defs`` entries referenced by name."""
    if node.kind == "union":
        return {"anyOf": [schema_of(n, defs) for n in node.args]}
    if node.kind == "none":
        return {"type": "null"}
    if node.kind == "callable":
        return {"type": "null", "description": "callables have no JSON form; only None is written"}
    if node.kind == "any":
        return {}
    if node.kind == "scalar":
        return {
            "float": _FLOAT_SCHEMA,
            "int": {"type": "integer"},
            "str": {"type": "string"},
            "bool": {"type": "boolean"},
            "complex": _COMPLEX_SCHEMA,
        }[node.name]
    if node.kind == "literal":
        return {"type": "string", "enum": list(node.args)}
    if node.kind == "alias":
        return {"type": "string", "description": f"the {node.name} literal"}
    if node.kind == "array":
        return _ARRAY_SCHEMA
    if node.kind == "tuple":
        if node.variadic:
            return {"type": "array", "items": schema_of(node.args[0], defs)}
        kinds = {(n.kind, n.name) for n in node.args}
        out: dict[str, Any] = {"type": "array", "minItems": len(node.args), "maxItems": len(node.args)}
        if len(kinds) == 1:
            out["items"] = schema_of(node.args[0], defs)
        return out
    if node.kind == "dict":
        return {"type": "object", "additionalProperties": schema_of(node.args[1], defs)}
    if node.kind == "cls":
        if node.name not in defs:
            cls = resolve(node.name)
            defs[node.name] = {}  # placeholder against recursion
            defs[node.name] = {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    f.name
                    for f in dataclasses.fields(cls)
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
                ],
                "properties": {
                    f.name: schema_of(parse(_annotation(f)), defs) for f in dataclasses.fields(cls)
                },
                "description": (cls.__doc__ or "").strip().split("\n")[0],
            }
        return {"$ref": f"#/$defs/{node.name}"}
    raise ValueError(f"schema: unhandled node {node.kind}")


def device_schema() -> dict[str, Any]:
    """The JSON schema of ``Device.to_dict()`` (``docs/schemas/device.schema.json``)."""
    defs: dict[str, Any] = {}
    body = schema_of(Node("cls", name="Device"), defs)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/splch/qutip-trap/docs/schemas/device.schema.json",
        "title": "qutip-trap Device",
        "description": "The device record of PLAN.md Section 3.3 as Device.to_dict() writes it (schema version 1).",
        "type": "object",
        "required": ["schema_version", "qutip_trap_version", "device_hash", "device"],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "const": SCHEMA_VERSION},
            "qutip_trap_version": {"type": "string"},
            "device_hash": {"type": "string", "description": "Device.hash() of the record written"},
            "device": body,
        },
        "$defs": defs,
    }
