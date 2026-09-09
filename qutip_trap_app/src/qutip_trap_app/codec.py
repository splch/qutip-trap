"""Dataclass <-> JSON codec with an array side-store: the serialization behind the run record (PLAN.md Section 14.3).

Section 14.3: "Records export to a JSON document with binary arrays (HDF5 optional) and re-import bitwise". The document is
the JSON encoding of the frozen dataclasses of :mod:`qutip_trap_app.record`; every ``numpy`` array is lifted out of the
document into an :data:`ArrayStore` under its path and replaced by a reference, so that the arrays travel as binary
(``.npy``, dtype and shape preserved, no pickling) and the document stays readable. Floats are written with Python's
shortest round-trip ``repr`` and read back as the same double; complex scalars are written as their two components;
dictionaries are written as ordered ``[key, value]`` pairs so that integer and tuple keys survive; tuples become lists and
are rebuilt from the field's type hint. Decoding is driven by the dataclass field annotations, so a record class needs no
per-class ``from_json``.

The digest of a record (:func:`digest`) is the SHA-256 of the canonical document bytes followed by every array's dtype,
shape and C-order bytes in key order: two records with the same digest are bitwise the same data, which is what the
Section 9.11 "record round trip" row asserts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

ArrayStore = dict[str, np.ndarray]
"""Array path -> array; the binary half of an exported record."""

JSON = Any
"""A JSON-compatible value (dict, list, str, int, float, bool, None)."""

_ARRAY_KEY = "__ndarray__"
_COMPLEX_KEY = "__complex__"
_ITEMS_KEY = "__items__"


class CodecError(ValueError):
    """A value the record codec cannot encode or decode."""


# ---- encoding ----------------------------------------------------------------------------------------------------------------


def encode(obj: object, arrays: ArrayStore, path: str = "") -> JSON:
    """The JSON form of ``obj``; arrays are moved into ``arrays`` under their path."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, complex):
        return {_COMPLEX_KEY: [float(obj.real), float(obj.imag)]}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise CodecError(f"{path}: object arrays are not serializable")
        if path in arrays:
            raise CodecError(f"{path}: duplicate array path")
        arrays[path] = np.ascontiguousarray(obj)
        return {_ARRAY_KEY: path}
    if isinstance(obj, np.generic):
        return encode(obj.item(), arrays, path)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: encode(getattr(obj, f.name), arrays, f"{path}/{f.name}" if path else f.name)
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        items = []
        for k, (key, value) in enumerate(obj.items()):
            items.append(
                [
                    encode(key, arrays, f"{path}/k{k}"),
                    encode(value, arrays, f"{path}/{_key_path(key)}"),
                ]
            )
        return {_ITEMS_KEY: items}
    if isinstance(obj, (tuple, list, frozenset, set)):
        seq = sorted(obj, key=repr) if isinstance(obj, (frozenset, set)) else list(obj)
        return [encode(v, arrays, f"{path}/{i}") for i, v in enumerate(seq)]
    raise CodecError(f"{path}: no JSON encoding for {type(obj).__qualname__}")


def _key_path(key: object) -> str:
    text = repr(key) if not isinstance(key, str) else key
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


# ---- decoding ----------------------------------------------------------------------------------------------------------------


def decode(hint: Any, data: JSON, arrays: ArrayStore, path: str = "") -> Any:
    """Rebuild the value of type ``hint`` from its JSON form and the array store."""
    origin = get_origin(hint)
    if hint is Any:
        return data
    if origin is Union or origin is types.UnionType:
        options = [a for a in get_args(hint) if a is not type(None)]
        if data is None:
            return None
        if len(options) == 1:
            return decode(options[0], data, arrays, path)
        return _decode_union(options, data, arrays, path)
    if origin is Literal:
        return data
    if hint is np.ndarray or origin is np.ndarray:
        if not (isinstance(data, dict) and _ARRAY_KEY in data):
            raise CodecError(f"{path}: expected an array reference")
        key = data[_ARRAY_KEY]
        if key not in arrays:
            raise CodecError(f"{path}: array {key!r} missing from the store")
        return arrays[key]
    if hint is complex:
        if isinstance(data, dict) and _COMPLEX_KEY in data:
            re, im = data[_COMPLEX_KEY]
            return complex(float(re), float(im))
        return complex(data)
    if hint is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise CodecError(f"{path}: expected a float, got {data!r}")
        return float(data)
    if hint is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise CodecError(f"{path}: expected an int, got {data!r}")
        return int(data)
    if hint is bool:
        if not isinstance(data, bool):
            raise CodecError(f"{path}: expected a bool, got {data!r}")
        return data
    if hint is str:
        if not isinstance(data, str):
            raise CodecError(f"{path}: expected a str, got {data!r}")
        return data
    if hint is type(None):
        return None
    if origin in (tuple, list):
        if not isinstance(data, list):
            raise CodecError(f"{path}: expected a list, got {type(data).__name__}")
        args = get_args(hint)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], v, arrays, f"{path}/{i}") for i, v in enumerate(data))
        if origin is tuple:
            if len(args) != len(data):
                raise CodecError(f"{path}: expected {len(args)} items, got {len(data)}")
            return tuple(decode(a, v, arrays, f"{path}/{i}") for i, (a, v) in enumerate(zip(args, data)))
        (item,) = args
        return [decode(item, v, arrays, f"{path}/{i}") for i, v in enumerate(data)]
    if origin in (frozenset, set):
        (item,) = get_args(hint)
        values = [decode(item, v, arrays, f"{path}/{i}") for i, v in enumerate(data)]
        return frozenset(values) if origin is frozenset else set(values)
    if origin is dict:
        if not (isinstance(data, dict) and _ITEMS_KEY in data):
            raise CodecError(f"{path}: expected an items mapping")
        k_hint, v_hint = get_args(hint)
        out: dict[Any, Any] = {}
        for k, (key, value) in enumerate(data[_ITEMS_KEY]):
            kk = decode(k_hint, key, arrays, f"{path}/k{k}")
            out[kk] = decode(v_hint, value, arrays, f"{path}/{_key_path(kk)}")
        return out
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(data, dict):
            raise CodecError(f"{path}: expected an object for {hint.__qualname__}")
        hints = get_type_hints(hint)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(hint):
            if f.name not in data:
                if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
                    continue
                raise CodecError(f"{path}: field {f.name!r} of {hint.__qualname__} missing")
            kwargs[f.name] = decode(
                hints[f.name], data[f.name], arrays, f"{path}/{f.name}" if path else f.name
            )
        return hint(**kwargs)
    raise CodecError(f"{path}: no decoder for type hint {hint!r}")


def _decode_union(options: list[Any], data: JSON, arrays: ArrayStore, path: str) -> Any:
    """A union of several non-None members: the record schema uses ``float | int``-free unions only where the JSON form
    identifies the member (a dataclass object, an array reference, a scalar)."""
    for opt in options:
        try:
            return decode(opt, data, arrays, path)
        except CodecError:
            continue
    raise CodecError(f"{path}: no member of {options!r} decodes {data!r}")


# ---- documents and digests -------------------------------------------------------------------------------------------------------


def dumps(document: JSON) -> bytes:
    """The canonical document bytes: sorted keys, no whitespace, NaN and infinities spelled as JSON's extensions."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=True, ensure_ascii=True
    ).encode("utf-8")


def loads(data: bytes) -> JSON:
    return json.loads(data.decode("utf-8"))


def digest(document_bytes: bytes, arrays: ArrayStore) -> str:
    """SHA-256 over the document and every array (dtype, shape, C-order bytes) in key order."""
    h = hashlib.sha256()
    h.update(document_bytes)
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(f"|{key}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def to_document(obj: object) -> tuple[bytes, ArrayStore]:
    """Encode a dataclass instance to (document bytes, array store)."""
    arrays: ArrayStore = {}
    doc = encode(obj, arrays)
    return dumps(doc), arrays


def from_document[T](cls: type[T], document_bytes: bytes, arrays: ArrayStore) -> T:
    """Decode a dataclass instance of ``cls`` from its document bytes and array store."""
    value: T = decode(cls, loads(document_bytes), arrays)
    return value


__all__ = [
    "JSON",
    "ArrayStore",
    "CodecError",
    "decode",
    "digest",
    "dumps",
    "encode",
    "from_document",
    "loads",
    "to_document",
]
