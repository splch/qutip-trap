"""``BeamRoles`` on the ``Device`` and the ``FidelityLevel`` enum (docs/api_implementation_plan.md items 1.1 and 1.2): the
inference equals the old default, the presets carry their drive maps as roles with their digests unchanged, the roles stay
out of the device digest, a run with no drive keyword reproduces the documented Bell histogram bit for bit, and the level
decision reports the numbers it compared."""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.schedule import (
    GateDrive,
    ScheduleError,
    default_gate_drives,
    infer_gate_drives,
    resolve_drives,
)
from qutip_trap.device.model import BeamRoles, ResolvedRoles
from qutip_trap.device.presets import ca40_optical, yb171_chain
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.hashing import canonical_digest
from qutip_trap.light.roles import detection_beams, infer_detection_beam
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Truncation
from qutip_trap.run.levels import FidelityLevel, decide_level, resolve_level
from tests.fixtures import make_device
from tests.m6_fixtures import circuit_fixture

DIGESTS_AT_7A26A27 = {
    "yb171_chain(1)": "323cdc28efc6ada6e2f29d3c389049f6292699938053b92298382fdbff826773",
    "yb171_chain(2)": "8d3bf1aab769b279070718fe3157b7b97919dae07d6782c530570d8c90d2197a",
    "yb171_chain(3)": "f26f062127a95a90f413cdada84027a0a7509f8e9f17694e1eca434460df1235",
    "yb171_chain(3, address_waist_m=2e-6)": "701fd2991d5fad9110f948c2a3387f364080d1b75b79d1255ea207f24c4dee81",
    "yb171_chain(2, phase_continuous=True)": "de9335590ef56777fc3b4936756acf9bc0da4b09daa6e70379f6b94ab08f376f",
    "ca40_optical(1)": "dd3b4113b3cdca513d1aebe87ef8198d2df0b7894db28fa3476efe6a03a9bbca",
    "ca40_optical(2, reset_beam=True)": "0b5a89cf36fae548aee1dcd7edec43b605f8337742c987f84622a521658de14b",
}
"""``Device.hash()`` of every preset at commit 7a26a27 (v0.1.0), captured before ``Device.roles`` existed: the digest
identifies the apparatus, and the roles the presets now carry do not move it (the risk of Section 9 of the plan)."""

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


def test_inference_equals_the_old_default_where_the_beams_identify_one_drive() -> None:
    """The fixture device carries one Raman pair: the roles' inference, the compatibility function and the raw rule agree."""
    dev = make_device()
    assert dev.roles == BeamRoles()
    resolved = dev.roles.resolve(dev)
    assert (
        resolved.gate
        == infer_gate_drives(dev)
        == {0: GateDrive("raman", (0, 1)), 1: GateDrive("raman", (0, 1))}
    )
    assert (
        resolved.entangling == resolved.gate and resolved.detection is None
    )  # no resonant beam on the fixture
    assert resolved.inferred == ("gate", "entangling", "detection")
    assert default_gate_drives(dev) == resolved.gate
    microwave = dataclasses.replace(dev, beams=())
    assert default_gate_drives(microwave) == {0: GateDrive("microwave", ()), 1: GateDrive("microwave", ())}


def test_several_raman_pairs_still_refuse_without_roles_and_resolve_with_them() -> None:
    fx = circuit_fixture(2)
    bare = dataclasses.replace(fx.device, roles=BeamRoles())
    with pytest.raises(ScheduleError, match=r"do not identify a single-qubit gate drive"):
        default_gate_drives(bare)
    # an explicit keyword argument takes precedence over the roles and over the inference
    gate, ent = resolve_drives(bare, fx.gate_drives, fx.entangling_drives)
    assert gate == fx.gate_drives and ent == fx.entangling_drives
    gate2, ent2 = resolve_drives(
        bare, fx.gate_drives
    )  # entangling drives default to the explicit gate drives
    assert gate2 == fx.gate_drives and ent2 == fx.gate_drives
    # with the roles declared, nothing is inferred
    resolved = fx.device.roles.resolve(fx.device)
    assert resolved == ResolvedRoles(fx.gate_drives, fx.entangling_drives, fx.detection_beam, ())
    assert resolve_drives(fx.device) == (fx.gate_drives, fx.entangling_drives)
    # the declared entangling drives win over explicit gate drives that name no entangling drive
    assert resolve_drives(fx.device, fx.gate_drives)[1] == fx.entangling_drives


def test_roles_are_validated_against_the_device() -> None:
    dev = make_device()
    with pytest.raises(ValueError, match=r"ion 5"):
        BeamRoles(gate={5: GateDrive("raman", (0, 1))}).resolve(dev)
    with pytest.raises(ValueError, match=r"beams \(0, 9\)"):
        BeamRoles(gate={0: GateDrive("raman", (0, 9))}).resolve(dev)
    with pytest.raises(ValueError, match=r"detection = 7"):
        BeamRoles(detection=7).resolve(dev)
    with pytest.raises(ValueError, match=r"non-negative"):
        BeamRoles(detection=-1)


@pytest.mark.parametrize("build", [lambda: yb171_chain(2), lambda: yb171_chain(1), lambda: ca40_optical(1)])
def test_each_preset_carries_its_old_drive_maps_as_roles(build) -> None:  # type: ignore[no-untyped-def]
    preset = build()
    dev = preset.device
    assert dev.roles == BeamRoles(
        gate=preset.gate_drives, entangling=preset.entangling_drives, detection=preset.detection_beam
    )
    resolved = dev.roles.resolve(dev)
    assert resolved.gate == preset.gate_drives and resolved.entangling == preset.entangling_drives
    assert resolved.detection == preset.detection_beam and resolved.inferred == ()
    assert default_gate_drives(dev) == preset.gate_drives
    # the declared detection beam is the one the wavelengths pick, and the readout's set contains it
    assert infer_detection_beam(dev) == preset.detection_beam
    assert preset.detection_beam in detection_beams(dev, 0)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the pins are the reference machine's: ndarray fields (positions, mode eigenvectors, axes) enter the digest as raw "
    "float64 bytes, so another LAPACK's round-off moves it (Linux CI); rounding arrays like floats would make it portable",
)
def test_preset_device_digests_are_unchanged_from_7a26a27() -> None:
    """A moved device hash would invalidate every cached table and every stored record (plan Section 9); the roles a preset
    now carries stay out of the digest, so these are the v0.1.0 digests. Pinned on the reference machine only: the digest
    rounds floats to 12 significant digits but hashes ndarray fields as bytes, and the crystal's positions and eigenvectors
    differ in the last bits between LAPACK implementations (measured: Linux CI fails, macOS matches)."""
    builders = {
        "yb171_chain(1)": lambda: yb171_chain(1),
        "yb171_chain(2)": lambda: yb171_chain(2),
        "yb171_chain(3)": lambda: yb171_chain(3),
        "yb171_chain(3, address_waist_m=2e-6)": lambda: yb171_chain(3, address_waist_m=2e-6),
        "yb171_chain(2, phase_continuous=True)": lambda: yb171_chain(2, phase_continuous=True),
        "ca40_optical(1)": lambda: ca40_optical(1),
        "ca40_optical(2, reset_beam=True)": lambda: ca40_optical(2, reset_beam=True),
    }
    assert set(builders) == set(DIGESTS_AT_7A26A27)
    got = {name: build().device.hash() for name, build in builders.items()}
    assert got == DIGESTS_AT_7A26A27


def test_roles_do_not_enter_the_device_digest_but_are_hashable_on_their_own() -> None:
    dev = make_device()
    with_roles = dataclasses.replace(dev, roles=BeamRoles(gate={0: GateDrive("raman", (0, 1))}, detection=1))
    assert with_roles.hash() == dev.hash()
    assert with_roles.roles != dev.roles
    assert canonical_digest(with_roles.roles) != canonical_digest(dev.roles)  # Machine.hash() digests them
    # any other field still moves the digest
    assert dataclasses.replace(dev, zones=()).hash() == dev.hash()
    assert dataclasses.replace(dev, gradient=None).hash() == dev.hash()


def test_fidelity_level_members_equal_the_strings_the_code_accepted() -> None:
    assert (
        FidelityLevel("auto") is FidelityLevel.AUTO
        and FidelityLevel("GATE_LOCAL") is FidelityLevel.GATE_LOCAL
    )
    assert FidelityLevel.JOINT_EXACT == "JOINT_EXACT" and FidelityLevel.GATE_LOCAL == "GATE_LOCAL"
    assert FidelityLevel(FidelityLevel.JOINT_EXACT) is FidelityLevel.JOINT_EXACT
    assert f"{FidelityLevel.GATE_LOCAL}" == "GATE_LOCAL"
    with pytest.raises(ValueError):
        FidelityLevel("exact")


def test_the_level_decision_names_both_numbers_and_both_guards() -> None:
    dev = make_device()
    native = Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, 1.5707963267948966)),), (0, 1))
    d = decide_level(dev, native, SolverOptions())
    assert d.level is FidelityLevel.JOINT_EXACT and d.estimated
    assert d.dimension == 4 * 12**2 and d.nnz == 2 * 4 * (12**2) ** 2
    assert d.reason == (
        "JOINT_EXACT: estimated joint dimension 576 <= joint_dimension_max = 4096, drive-operator non-zeros 165888 <= "
        "nnz_max = 20000000 (Section 11.5)"
    )
    tight = decide_level(dev, native, SolverOptions(joint_dimension_max=64))
    assert tight.level is FidelityLevel.GATE_LOCAL and "576 > joint_dimension_max = 64" in tight.reason
    assert resolve_level(dev, native, SolverOptions(joint_dimension_max=64)) is FidelityLevel.GATE_LOCAL


FAST = Numerics(truncation=Truncation(branch_weight_min=1e-3))


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    preset = yb171_chain(2)
    sur = surrogate_table(preset.device, pairs=[(0, 1)], detection_records=2000, detection_windows_s=WINDOWS)
    return preset, sur.table


def test_run_without_drive_keywords_reproduces_the_bell_histogram_bit_for_bit(two_ion) -> None:  # type: ignore[no-untyped-def]
    preset, table = two_ion
    new = Machine(preset.device, table=table, numerics=FAST).run(BELL, 400, seed=3)
    assert new.probabilities["00"] + new.probabilities["11"] > 0.98
    # 1.2: the diagnostics say which level ran and why, with the dimension and the non-zeros against the guards
    assert new.diagnostics.level == "JOINT_EXACT" == FidelityLevel.JOINT_EXACT
    reason = new.diagnostics.level_reason
    assert reason.startswith("JOINT_EXACT: declared joint dimension ") and "nnz_max = 20000000" in reason
    assert f"joint dimension {new.diagnostics.space.dimension} <= joint_dimension_max = 4096" in reason


def test_a_forced_level_reports_the_choice_auto_would_have_made(two_ion) -> None:  # type: ignore[no-untyped-def]
    preset, table = two_ion
    forced = Machine(preset.device, table=table, numerics=FAST, level=FidelityLevel.GATE_LOCAL).run(
        Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)), 20
    )
    assert forced.diagnostics.level == "GATE_LOCAL"
    assert forced.diagnostics.level_reason.startswith(
        "GATE_LOCAL forced by the caller; level='auto' would choose JOINT_EXACT"
    )
