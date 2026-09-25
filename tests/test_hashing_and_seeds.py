"""The canonical digest behind Device.hash() and the keyed seeds (PLAN.md Section 3.4)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt
from hypothesis import given
from hypothesis import strategies as st

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.hashing import canonical_digest, canonical_float
from tests.fixtures import make_device

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Inner:
    arr: np.ndarray
    table: dict[str, float]


@dataclasses.dataclass(frozen=True)
class Outer:
    x: float
    inner: Inner
    fn: object
    label: str


def _fn(t: float) -> float:
    return 2.0 * t


def test_float_rounding_to_twelve_significant_digits() -> None:
    assert canonical_float(1.0) == canonical_float(1.0 + 1e-13)
    assert canonical_float(1.0) != canonical_float(1.0 + 1e-10)
    assert canonical_float(0.0) == canonical_float(-0.0)


def test_dict_order_and_array_layout_do_not_matter_but_values_do() -> None:
    a = Outer(1.5, Inner(np.arange(6.0).reshape(2, 3), {"b": 2.0, "a": 1.0}), _fn, "x")
    b = Outer(1.5, Inner(np.asfortranarray(np.arange(6.0).reshape(2, 3)), {"a": 1.0, "b": 2.0}), _fn, "x")
    assert canonical_digest(a) == canonical_digest(b)
    c = Outer(1.5, Inner(np.arange(6.0).reshape(3, 2), {"a": 1.0, "b": 2.0}), _fn, "x")
    assert canonical_digest(a) != canonical_digest(c), "a reshape changes identity"
    d = Outer(1.5000001, Inner(np.arange(6.0).reshape(2, 3), {"a": 1.0, "b": 2.0}), _fn, "x")
    assert canonical_digest(a) != canonical_digest(d)


def test_qobj_fields_are_excluded() -> None:
    @dataclasses.dataclass(frozen=True)
    class Holder:
        q: object
        n: int

    assert canonical_digest(Holder(qt.sigmax(), 3)) == canonical_digest(Holder(qt.sigmaz(), 3))
    assert canonical_digest(Holder(qt.QobjEvo(qt.sigmax()), 3)) == canonical_digest(Holder(qt.sigmaz(), 3))
    assert canonical_digest(Holder(qt.sigmax(), 3)) != canonical_digest(Holder(qt.sigmax(), 4))


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=5), st.floats(allow_nan=False, allow_infinity=False), max_size=6
    )
)
def test_digest_is_insertion_order_independent(table: dict[str, float]) -> None:
    reversed_table = dict(reversed(list(table.items())))
    assert canonical_digest(table) == canonical_digest(reversed_table)


def test_device_hash_is_stable_across_processes() -> None:
    here = make_device().hash()
    code = "from tests.fixtures import make_device; print(make_device().hash())"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == here
    assert len(here) == 64


def test_seed_children_are_keyed_and_order_independent() -> None:
    spec = SeedSpec(root=12345)
    a = np.random.default_rng(spec.child(0, 1, 2, 0, "heating")).random(3)
    b = np.random.default_rng(spec.child(0, 1, 2, 0, "heating")).random(3)
    c = np.random.default_rng(spec.child(0, 1, 2, 0, "dephasing")).random(3)
    d = np.random.default_rng(spec.child(0, 1, 3, 0, "heating")).random(3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d), "the shot slot matters (Section 3.4)"


def test_seed_channel_key_is_deterministic_across_processes() -> None:
    code = "from qutip_trap.dynamics.engine import SeedSpec; print(SeedSpec.channel_key('photon_count'))"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert int(proc.stdout.strip()) == SeedSpec.channel_key("photon_count")


def test_seed_spec_rejects_negative_keys() -> None:
    with pytest.raises(ValueError):
        SeedSpec(root=-1)
    with pytest.raises(ValueError):
        SeedSpec(root=1).child(0, -1, 0, 0, "x")
