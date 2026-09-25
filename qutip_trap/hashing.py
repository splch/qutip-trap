"""Canonical serialization and SHA-256 digest of the frozen data objects: ``Device.hash()`` and the cache keys.

Python's ``hash()`` is no identity here (the records carry ndarrays, dicts and Qobj), so the digest is taken over a
canonical form: dataclass fields in declaration order, floats to 12 significant digits, ndarrays as dtype,
shape and C-order bytes, mappings and sets sorted, callables by qualified name plus a digest of their source, Qobj fields
excluded. A field with ``metadata={"hash": "exclude"}`` is left out (``Device.roles``, which ``Machine.hash()`` carries),
one with ``metadata={"hash": "skip_default"}`` while it holds its default (``CalEntry.kind``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import qutip as qt

FLOAT_SIGNIFICANT_DIGITS = 12


def canonical_float(x: float) -> str:
    """A float rounded to 12 significant digits, as text; nan/inf spelled out."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0.0:
        return "0"
    return format(x, f".{FLOAT_SIGNIFICANT_DIGITS - 1}e")


def _callable_token(fn: Callable[..., Any]) -> str:
    """Module, qualname and a digest of the source; a callable without readable source (a REPL lambda, a builtin) is
    keyed by its identity too, since two such callables can share a qualname: the key is then process-local."""
    name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return f"callable:{name}:no-source:id={id(fn):x}"
    return f"callable:{name}:{hashlib.sha256(src.encode()).hexdigest()}"


def _feed(h: Any, obj: object) -> None:
    """Write the canonical form of ``obj`` into the hash object ``h``."""
    if obj is None:
        h.update(b"N;")
    elif isinstance(obj, bool):
        h.update(b"T;" if obj else b"F;")
    elif isinstance(obj, int):
        h.update(f"i{obj};".encode())
    elif isinstance(obj, float):
        h.update(f"f{canonical_float(obj)};".encode())
    elif isinstance(obj, complex):
        h.update(f"c{canonical_float(obj.real)},{canonical_float(obj.imag)};".encode())
    elif isinstance(obj, str):
        h.update(f"s{len(obj)}:".encode())
        h.update(obj.encode())
        h.update(b";")
    elif isinstance(obj, bytes):
        h.update(f"b{len(obj)}:".encode())
        h.update(obj)
        h.update(b";")
    elif isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        h.update(f"a{arr.dtype.str}{arr.shape}:".encode())
        h.update(arr.tobytes())
        h.update(b";")
    elif isinstance(obj, np.generic):
        _feed(h, obj.item())
    elif isinstance(obj, (qt.Qobj, qt.QobjEvo)):
        h.update(b"Q;")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"D{type(obj).__qualname__}(".encode())
        for f in dataclasses.fields(obj):
            rule = f.metadata.get("hash")
            if rule == "exclude":
                continue
            value = getattr(obj, f.name)
            if rule == "skip_default" and f.default is not dataclasses.MISSING and value == f.default:
                continue
            h.update(f"{f.name}=".encode())
            _feed(h, value)
        h.update(b");")
    elif isinstance(obj, Mapping):
        h.update(f"m{len(obj)}(".encode())
        for key, value in sorted(obj.items(), key=lambda kv: repr(kv[0])):
            _feed(h, key)
            h.update(b"->")
            _feed(h, value)
        h.update(b");")
    elif isinstance(obj, (frozenset, set)):
        h.update(f"S{len(obj)}(".encode())
        for item in sorted(obj, key=repr):
            _feed(h, item)
        h.update(b");")
    elif isinstance(obj, (tuple, list)):
        h.update(f"t{len(obj)}(".encode())
        for item in obj:
            _feed(h, item)
        h.update(b");")
    elif callable(obj):
        h.update(_callable_token(obj).encode())
        h.update(b";")
    else:
        raise TypeError(f"no canonical serialization for {type(obj).__qualname__}")


def canonical_digest(obj: object) -> str:
    """The SHA-256 hex digest of the canonical serialization of ``obj``."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()
