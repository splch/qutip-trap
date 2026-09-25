"""Dataclass -> JSON encoding with an array side-store, behind the record's digest.

Every ``numpy`` array is lifted out of the document into an :data:`ArrayStore` under its path and replaced by a reference;
floats keep Python's shortest round-trip ``repr``, complex scalars are written as their two components, dictionaries as
ordered ``[key, value]`` pairs (so integer and tuple keys survive) and tuples as lists. The digest is the SHA-256 of the canonical document bytes and every array's dtype, shape and bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

ArrayStore = dict[str, np.ndarray]
"""Array path -> array."""

JSON = Any
"""A JSON-compatible value (dict, list, str, int, float, bool, None)."""

_ARRAY_KEY = "__ndarray__"
_COMPLEX_KEY = "__complex__"
_ITEMS_KEY = "__items__"


class CodecError(ValueError):
    """A value the record codec cannot encode."""


def encode(obj: object, arrays: ArrayStore, path: str = "") -> JSON:
    """The JSON form of ``obj``; arrays are moved into ``arrays`` under their path."""
    if obj is None or isinstance(obj, (bool, int, str, float)):
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
        return {
            _ITEMS_KEY: [
                [encode(key, arrays, f"{path}/k{k}"), encode(value, arrays, f"{path}/{_key_path(key)}")]
                for k, (key, value) in enumerate(obj.items())
            ]
        }
    if isinstance(obj, (tuple, list)):
        return [encode(v, arrays, f"{path}/{i}") for i, v in enumerate(obj)]
    raise CodecError(f"{path}: no JSON encoding for {type(obj).__qualname__}")


def _key_path(key: object) -> str:
    text = repr(key) if not isinstance(key, str) else key
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def dumps(document: JSON) -> bytes:
    """The canonical document bytes: sorted keys, no whitespace, NaN and infinities spelled as JSON's extensions."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=True, ensure_ascii=True
    ).encode("utf-8")


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
    return dumps(encode(obj, arrays)), arrays
