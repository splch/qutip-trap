"""The deprecation helper of docs/api_implementation_plan.md item 0.1 (``qutip_trap/_compat.py``): the deadline regex, the
warning category and message, the rewrite producing identical results, the attribution to the call site, and the test
session's error filter, which makes a deprecated path used inside the package fail CI."""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import pytest

from qutip_trap._compat import (
    QutipTrapDeprecationWarning,
    deprecated,
    deprecated_alias,
    deprecated_parameter,
    validate,
    warn,
)

FIX = "Call the new form instead."


def test_the_category_is_a_deprecation_warning() -> None:
    assert issubclass(QutipTrapDeprecationWarning, DeprecationWarning)


@pytest.mark.parametrize("deadline", ["0.4", "v0.4.1", "v0", "0.4.0", "V0.4", "v0.4 ", "vx.y"])
def test_the_deadline_must_be_a_release(deadline: str) -> None:
    with pytest.raises(ValueError, match=r"deadline must be a release"):
        validate(deadline, FIX)
    with pytest.raises(ValueError, match=r"deadline must be a release"):
        deprecated(deadline=deadline, fix=FIX)


@pytest.mark.parametrize("deadline", ["v0.4", "v1.0", "v12.34"])
def test_a_release_deadline_is_accepted(deadline: str) -> None:
    validate(deadline, FIX)


@pytest.mark.parametrize("fix", ["", "use Machine.run instead.", "Use Machine.run instead", "Fix.", "   "])
def test_the_fix_must_be_a_full_sentence(fix: str) -> None:
    with pytest.raises(ValueError, match=r"fix must be a full sentence"):
        validate("v0.4", fix)


@pytest.mark.parametrize(
    "fix", ["Use Machine.run instead.", "`Machine.run` replaces it.", "Pass roles= on the Device!"]
)
def test_a_full_sentence_fix_is_accepted(fix: str) -> None:
    validate("v0.4", fix)


def _plain(x: int, *, y: int = 1) -> int:
    """The undecorated form."""
    return 10 * x + y


old_plain = deprecated(deadline="v0.4", fix=FIX)(_plain)


def test_a_deprecated_function_warns_with_the_deadline_and_the_fix_and_keeps_its_result() -> None:
    with pytest.warns(
        QutipTrapDeprecationWarning,
        match=r"tests\.test_compat\._plain is deprecated and will be removed in qutip-trap v0\.4 at the earliest\. "
        + FIX.replace(".", r"\."),
    ):
        assert old_plain(3, y=4) == _plain(3, y=4) == 34
    assert inspect.signature(old_plain) == inspect.signature(_plain)
    assert old_plain.__doc__ == _plain.__doc__
    assert old_plain.__deprecated__.startswith(
        "tests.test_compat._plain is deprecated"
    )  # PEP 702, for type checkers


@deprecated(deadline="v0.4", fix="Construct the New record instead.")
@dataclasses.dataclass(frozen=True)
class OldRecord:
    """A frozen dataclass under the class decorator."""

    a: int
    b: float = 2.0


def test_a_deprecated_class_warns_on_construction_and_stays_a_frozen_dataclass() -> None:
    with pytest.warns(QutipTrapDeprecationWarning, match=r"tests\.test_compat\.OldRecord is deprecated"):
        rec = OldRecord(1)
    assert (rec.a, rec.b) == (1, 2.0)
    with pytest.warns(QutipTrapDeprecationWarning):
        assert dataclasses.replace(rec, b=3.0) == OldRecord(1, 3.0)
    assert [f.name for f in dataclasses.fields(OldRecord)] == ["a", "b"]
    assert OldRecord.__dataclass_params__.frozen
    assert list(inspect.signature(OldRecord).parameters) == ["a", "b"]
    assert OldRecord.__deprecated__.startswith("tests.test_compat.OldRecord is deprecated")


def _drop_old(kwargs: dict[str, object]) -> dict[str, object]:
    """The rewrite: ``old`` becomes ``new``."""
    return {**{k: v for k, v in kwargs.items() if k != "old"}, "new": kwargs["old"]}


@deprecated_parameter(name="old", deadline="v0.4", fix="Pass new= instead.", rewrite=_drop_old)
def takes_new(*, new: int = 0, other: int = 0) -> tuple[int, int]:
    """A function whose ``old`` keyword was renamed ``new``."""
    return new, other


def test_a_deprecated_parameter_is_rewritten_to_an_identical_result() -> None:
    with pytest.warns(
        QutipTrapDeprecationWarning,
        match=r"the 'old' argument of tests\.test_compat\.takes_new is deprecated .* v0\.4 .* Pass new= instead\.",
    ):
        rewritten = takes_new(old=7, other=1)
    assert rewritten == takes_new(new=7, other=1) == (7, 1)
    assert takes_new.__doc__ == "A function whose ``old`` keyword was renamed ``new``."


def test_a_call_without_the_deprecated_parameter_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert takes_new(new=2) == (2, 0)


def test_a_rewrite_that_keeps_the_old_name_is_refused() -> None:
    @deprecated_parameter(name="old", deadline="v0.4", fix="Pass new= instead.", rewrite=lambda kw: dict(kw))
    def f(**kwargs: int) -> dict[str, int]:
        return kwargs

    with pytest.warns(QutipTrapDeprecationWarning), pytest.raises(ValueError, match=r"must remove it"):
        f(old=1)


def _drop_older(kwargs: dict[str, object]) -> dict[str, object]:
    return {**{k: v for k, v in kwargs.items() if k != "older"}, "other": kwargs["older"]}


@deprecated_parameter(name="old", deadline="v0.4", fix="Pass new= instead.", rewrite=_drop_old)
@deprecated_parameter(name="older", deadline="v0.4", fix="Pass other= instead.", rewrite=_drop_older)
def takes_two(*, new: int = 0, other: int = 0) -> tuple[int, int]:
    return new, other


def test_stacked_decorators_attribute_every_warning_to_the_call_site() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert takes_two(old=1, older=2) == (1, 2)
        line = inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]
    assert sorted(str(w.message)[:24] for w in record) == [
        "the 'old' argument of te",
        "the 'older' argument of ",
    ]
    assert {(w.filename, w.lineno) for w in record} == {(__file__, line)}, [
        (w.filename, w.lineno) for w in record
    ]


def test_a_call_site_warns_once_under_the_default_filter() -> None:
    """Python's ``default`` action keys on (message, category, module, line): the same site warns once however often it
    runs, a second site warns again."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(3):
            takes_new(old=1)
        first = inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]
        takes_new(old=1)
        second = inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]
    assert [w.lineno for w in record] == [first, second]


def _renamed(x: int) -> int:
    """The new name."""
    return x + 1


old_name = deprecated_alias(
    _renamed, deadline="v0.4", fix="Call _renamed() instead.", name="tests.test_compat.old_name"
)


class Box:
    def new(self, x: int) -> int:
        """The new method."""
        return x + 100

    old = deprecated_alias(new, deadline="v0.4", fix="Call Box.new() instead.")


def test_a_deprecated_alias_names_itself_and_forwards() -> None:
    with pytest.warns(
        QutipTrapDeprecationWarning, match=r"tests\.test_compat\.old_name is deprecated .* v0\.4"
    ):
        assert old_name(1) == 2
    assert old_name.__wrapped__ is _renamed
    assert inspect.signature(old_name) == inspect.signature(_renamed)
    assert old_name.__doc__ == "Deprecated alias of ``tests.test_compat._renamed``. Call _renamed() instead."
    assert old_name.__deprecated__.startswith("tests.test_compat.old_name is deprecated")


def test_a_deprecated_alias_binds_as_a_method_and_learns_its_name_from_the_class() -> None:
    box = Box()
    with pytest.warns(
        QutipTrapDeprecationWarning, match=r"tests\.test_compat\.Box\.old is deprecated .* Box\.new\(\)"
    ):
        assert box.old(1) == 101
    with pytest.warns(QutipTrapDeprecationWarning):
        assert Box.old(box, 2) == 102  # unbound use too
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        box.old(0)
        line = inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]
    assert [(w.filename, w.lineno) for w in record] == [(__file__, line)]


def test_warn_can_point_past_the_method_that_calls_it() -> None:
    def method() -> None:
        warn("Method is deprecated.", stacklevel=2)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        method()
        line = inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]
    assert [(w.filename, w.lineno) for w in record] == [(__file__, line)]


def test_the_test_session_turns_the_warning_into_an_error() -> None:
    """``filterwarnings = error::qutip_trap._compat.QutipTrapDeprecationWarning`` in pyproject.toml: a deprecated path used
    inside the package (or a test that forgets ``pytest.warns``) fails, which is item 0.1's guard for Phase 2's rewrites."""
    with pytest.raises(QutipTrapDeprecationWarning):
        warnings.warn("a deprecated path", QutipTrapDeprecationWarning, stacklevel=1)
