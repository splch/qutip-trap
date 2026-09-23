"""Canonical serialization and SHA-256 digest of the frozen data objects (the identity behind ``Device.hash()``).

Dataclass fields go in declaration order, floats rounded to 12 significant digits, ndarrays as dtype, shape and C-order
bytes, dicts by sorted key, callables by qualified name plus a source digest. Qobj fields and fields marked
``metadata={"hash": "exclude"}`` are left out; ``"skip_default"`` fields enter only when they differ from their default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

FLOAT_SIGNIFICANT_DIGITS = 12


def _is_qobj(value: object) -> bool:
    module = type(value).__module__ or ""
    return module == "qutip" or module.startswith("qutip.")


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
    """A token for a callable: module, qualname and a digest of its source, plus ``id(fn)`` when the source cannot be
    read (a REPL lambda's qualname is no fingerprint; a process-local key beats a wrong cache hit)."""
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
    elif _is_qobj(obj):
        h.update(b"Q;")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"D{type(obj).__qualname__}(".encode())
        for f in dataclasses.fields(obj):
            rule = f.metadata.get("hash")
            if rule == "exclude":
                continue
            value = getattr(obj, f.name)
            if rule == "skip_default" and f.default is not dataclasses.MISSING and value == f.default:
                # a field added later enters only when set, so digests taken before it existed stay valid
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
