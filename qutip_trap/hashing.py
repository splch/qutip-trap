"""Canonical serialization and digest for the frozen data objects (PLAN.md Appendix E, preamble).

Identity for the calibration cache and the run record is ``Device.hash()``, a canonical serialization
rather than Python's ``hash()`` (which ``frozen=True`` does not supply for the ndarray, dict and Qobj
fields these objects carry). The rules, verbatim from Appendix E:

- dataclass fields enumerated in declaration order;
- floats rounded to 12 significant digits;
- ndarrays as dtype plus C-order bytes (the shape is included so that reshapes differ);
- dicts by sorted key;
- callables by qualified name plus a digest of their source;
- Qobj fields excluded;
- fields declared with ``field(metadata={"hash": "exclude"})`` left out: ``Device.roles`` (0.2.0), the operator's
  assignment of which beams play which part, which is not the apparatus the digest identifies, so every digest taken
  before roles existed stays valid (``Machine.hash()`` carries the roles instead).

A cross-process test asserts that the same device yields the same digest (tests/test_hashing.py).
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
    """A stable token for a coefficient function: module, qualname and a digest of its source.

    When ``inspect.getsource`` cannot read the source (a function defined in a REPL or an ``exec``, a builtin, a C
    extension) the qualname alone is NOT a fingerprint: two different lambdas defined interactively share
    ``<module>.<lambda>`` and would hash equal, so a device carrying one would silently reuse the other's calibration
    cache entry. Such a callable is therefore keyed by its identity as well, which makes the hash process-local and
    refuses to pretend otherwise (the caller sees a fresh cache key rather than a wrong hit)."""
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
        h.update(b"Q;")  # Qobj fields are excluded from identity (Appendix E)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"D{type(obj).__qualname__}(".encode())
        for f in dataclasses.fields(obj):
            if f.metadata.get("hash") == "exclude":
                continue
            h.update(f"{f.name}=".encode())
            _feed(h, getattr(obj, f.name))
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


__all__ = ["FLOAT_SIGNIFICANT_DIGITS", "canonical_digest", "canonical_float"]
