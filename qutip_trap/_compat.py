"""Deprecation helpers: the typed warning, the decorators and the policy behind them (docs/api_implementation_plan.md
Section 1 and item 0.1; docs/api_proposal.md Section 4.11).

Policy. Names are added first, deprecated second and removed third: a name is deprecated only once its replacement has
shipped in a release, and it is removed no earlier than two minor releases after the warning, always leaving a
warning-free path. Every warning names the deadline (``deadline="v0.4"``: the first release that MAY remove the name) and
the fix, a full sentence the reader can act on. ``docs/deprecations.md`` (from 0.2.0) lists every entry as "deprecated in /
removed in".

Mechanics. :class:`QutipTrapDeprecationWarning` is a ``DeprecationWarning``, so Python's default filters show it when the
deprecated call is made from ``__main__`` and hide it inside libraries, pytest shows it, and this package's own test
session turns it into an error (``filterwarnings`` in ``pyproject.toml``), so a deprecated path used inside the package
fails CI. Every warning is attributed to the caller's line (the frames of this module are skipped, however the decorators
are stacked), which under Python's ``default`` filter action means once per call site. The decorators follow Cirq's
``_compat`` (``deprecated(deadline, fix)``, ``deprecated_parameter`` with a rewrite of the call) and mark what they
decorate with ``__deprecated__`` (PEP 702), so a type checker reports the use as well.
"""

from __future__ import annotations

import functools
import re
import sys
import types
import warnings
from collections.abc import Callable
from types import FrameType
from typing import Any, Final, TypeVar, cast


class QutipTrapWarning(UserWarning):
    """The base of every warning this package issues on its own behalf: a truncation the run could not make exact
    (``hilbert.truncation.TruncationWarning``), a deprecated name (:class:`QutipTrapDeprecationWarning`). Filter on it to
    hear or silence the package as a whole; ``warnings.simplefilter("error", QutipTrapWarning)`` makes silence a test."""


class QutipTrapDeprecationWarning(QutipTrapWarning, DeprecationWarning):
    """A qutip-trap name or call form that will be removed; the message names the deadline and the fix."""


_DEADLINE: Final = re.compile(r"^v\d+\.\d+$")
_THIS_FILE: Final[str] = __file__
_T = TypeVar("_T", bound=Callable[..., object])


def validate(deadline: str, fix: str) -> None:
    """Refuse a deadline that is not a release (``^v\\d+\\.\\d+$``) or a fix that is not a full sentence.

    A full sentence starts with a capital letter or a quoted name and ends with a period, so that the warning reads as
    an instruction ("Use Machine.run instead.") rather than a fragment."""
    if not _DEADLINE.match(deadline):
        raise ValueError(f"deadline must be a release such as 'v0.4' (^v\\d+\\.\\d+$), got {deadline!r}")
    text = fix.strip()
    if not text or not (text[0].isupper() or text[0] in "`'\"") or text[-1] not in ".!?" or " " not in text:
        raise ValueError(
            "fix must be a full sentence (a capital letter or a quoted name first, a period last, more than one word), "
            f"got {fix!r}"
        )


def _qualified(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    return f"{module}.{name}" if module else str(name)


def message(what: str, deadline: str, fix: str) -> str:
    """The one sentence shape every deprecation warning of the package has: what, the deadline, the fix."""
    return f"{what} is deprecated and will be removed in qutip-trap {deadline} at the earliest. {fix}"


_message = message


def warn(message: str, *, stacklevel: int = 1) -> None:
    """Issue ``message`` as a :class:`QutipTrapDeprecationWarning` attributed to the first frame outside this module.

    ``stacklevel`` counts further frames of the caller to skip: 1 (the default) attributes the warning to the line that
    entered this module, 2 to that line's caller, which is what a method that warns on its own behalf passes."""
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename == _THIS_FILE:
        frame = frame.f_back
        level += 1
    warnings.warn(message, QutipTrapDeprecationWarning, stacklevel=level + stacklevel)


def deprecated(*, deadline: str, fix: str) -> Callable[[_T], _T]:
    """Mark a function, method or class deprecated: every call (every instantiation) warns once per call site, and the
    object carries ``__deprecated__`` for type checkers (PEP 702, through :func:`warnings.deprecated`).

        @deprecated(deadline="v0.4", fix="Call Machine.run instead.")
        def run_kwargs(self) -> dict[str, object]: ...
    """
    validate(deadline, fix)

    def decorate(obj: _T) -> _T:
        message = _message(_qualified(obj), deadline, fix)
        return warnings.deprecated(message, category=QutipTrapDeprecationWarning)(obj)

    return decorate


def deprecated_parameter[**P, R](
    *, name: str, deadline: str, fix: str, rewrite: Callable[[dict[str, Any]], dict[str, Any]]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark one keyword argument of a function deprecated and rewrite calls that pass it.

    When a call passes ``name`` as a keyword, the wrapper warns (once per call site) and calls ``rewrite`` with a copy of
    the call's keyword arguments; ``rewrite`` returns the keyword arguments to call with, ``name`` removed and its value
    moved to its new home, so the old call and the new call produce identical results. Positional use of the parameter
    is not detected: the arguments this decorator is for are keyword-only. Decorators stack, one per parameter.

        @deprecated_parameter(name="gate_drives", deadline="v0.5", fix="Set Device.roles instead.",
                              rewrite=lambda kw: {**{k: v for k, v in kw.items() if k != "gate_drives"}, ...})
        def run(circuit, device, shots, *, ...): ...
    """
    validate(deadline, fix)

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        message = _message(f"the {name!r} argument of {_qualified(func)}", deadline, fix)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if name not in kwargs:
                return func(*args, **kwargs)
            warn(message)
            rewritten = rewrite(dict(kwargs))
            if name in rewritten:
                raise ValueError(f"the rewrite for the deprecated argument {name!r} must remove it")
            # the rewrite changes the call's keyword arguments, so P no longer describes them
            return cast("Callable[..., R]", func)(*args, **rewritten)

        return wrapper

    return decorate


class _DeprecatedAlias[**P, R]:
    """The callable :func:`deprecated_alias` returns: warns, then calls the new name; on a class it binds like a method."""

    __deprecated__: str

    def __init__(self, new: Callable[P, R], *, deadline: str, fix: str, name: str | None) -> None:
        self._new = new
        self._deadline = deadline
        self._fix = fix
        self._name = name
        functools.update_wrapper(self, new)
        self.__doc__ = f"Deprecated alias of ``{_qualified(new)}``. {fix}"
        self.__deprecated__ = self.message

    def __set_name__(self, owner: type, name: str) -> None:
        if self._name is None:
            self._name = f"{owner.__module__}.{owner.__qualname__}.{name}"
            self.__deprecated__ = self.message

    @property
    def message(self) -> str:
        what = self._name if self._name is not None else f"this alias of {_qualified(self._new)}"
        return _message(what, self._deadline, self._fix)

    def __get__(self, instance: object, owner: type | None = None) -> Callable[..., R]:
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        warn(self.message)
        return self._new(*args, **kwargs)


def deprecated_alias[**P, R](
    new: Callable[P, R], *, deadline: str, fix: str, name: str | None = None
) -> Callable[P, R]:
    """The old name of a renamed function or method: a callable that warns once per call site and forwards to ``new``.

    In a class body the alias learns its own name (``Result.to_ionq_json``) when the class is created; at module level pass
    ``name`` (the dotted name the message should show), else the message names ``new``. The alias keeps ``new``'s signature
    (``__wrapped__``) and documents itself as the deprecated alias.

        class Result:
            def to_ionq_v1_probabilities(self) -> dict[str, float]: ...
            to_ionq_json = deprecated_alias(to_ionq_v1_probabilities, deadline="v0.4",
                                            fix="Call Result.to_ionq_v1_probabilities() instead.")
    """
    validate(deadline, fix)
    return _DeprecatedAlias(new, deadline=deadline, fix=fix, name=name)
