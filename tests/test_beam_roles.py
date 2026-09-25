"""``BeamRoles`` on the ``Device`` and the level decision: the inference where the beams identify one drive, the presets'
drive maps as roles, the roles out of the device digest, and a run that reports its level and why."""

from __future__ import annotations

import dataclasses

import pytest

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
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.hashing import canonical_digest
from qutip_trap.light.roles import detection_beams, infer_detection_beam
from qutip_trap.run.levels import FidelityLevel, decide_level, within_budget
from tests.fixtures import BELL, FAST, make_device, run, two_ion_surrogate


def test_inference_where_the_beams_identify_one_drive() -> None:
    """The fixture device carries one Raman pair: the roles' inference, the scheduler's default and the raw rule agree."""
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
    fx = yb171_chain(2)
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
def test_each_preset_carries_its_drive_maps_as_roles(build) -> None:  # type: ignore[no-untyped-def]
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


def test_roles_do_not_enter_the_device_digest_but_are_hashable_on_their_own() -> None:
    dev = make_device()
    with_roles = dataclasses.replace(dev, roles=BeamRoles(gate={0: GateDrive("raman", (0, 1))}, detection=1))
    assert with_roles.hash() == dev.hash()
    assert with_roles.roles != dev.roles
    assert canonical_digest(with_roles.roles) != canonical_digest(dev.roles)  # Machine.hash() digests them


def test_the_level_decision_names_both_numbers_and_both_guards() -> None:
    """Two ions with two resolved modes at d_m = 12: dimension 576 and N 2^N prod d_m^2 = 165888 drive non-zeros."""
    space = HilbertSpace(
        (2, 2), tuple(ModeTruncation(m, 12, (0, 4), 0.1) for m in (2, 3)), None, (0, 1, 4, 5)
    )
    budget = within_budget(space, SolverOptions())
    d = decide_level(budget, SolverOptions(), FidelityLevel.AUTO)
    assert d.level is FidelityLevel.JOINT_EXACT and d.inside and not d.forced
    assert d.dimension == 4 * 12**2 and d.nnz == 2 * 4 * (12**2) ** 2
    assert d.reason == (
        "JOINT_EXACT: declared joint dimension 576 <= joint_dimension_max = 4096, drive-operator non-zeros 165888 <= "
        "nnz_max = 20000000 (Section 11.5)"
    )
    tight_options = SolverOptions(joint_dimension_max=64)
    tight = decide_level(within_budget(space, tight_options), tight_options, FidelityLevel.AUTO)
    assert tight.level is FidelityLevel.GATE_LOCAL and "576 > joint_dimension_max = 64" in tight.reason
    forced = decide_level(budget, SolverOptions(), FidelityLevel.GATE_LOCAL)
    assert forced.level is FidelityLevel.GATE_LOCAL and forced.inside and forced.forced
    assert forced.reason == f"GATE_LOCAL forced by the caller; level='auto' would choose {d.reason}"


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    preset = yb171_chain(2)
    sur = two_ion_surrogate(2000)
    return preset, sur.table


def test_a_bell_run_reports_its_level_and_why(two_ion) -> None:  # type: ignore[no-untyped-def]
    preset, table = two_ion
    new = run(BELL, preset.device, 400, table=table, numerics=FAST, seed=3)
    assert new.probabilities["00"] + new.probabilities["11"] > 0.98
    # the diagnostics say which level ran and why, with the dimension and the non-zeros against the guards
    assert new.diagnostics.level == "JOINT_EXACT" == FidelityLevel.JOINT_EXACT
    reason = new.diagnostics.level_reason
    assert reason.startswith("JOINT_EXACT: declared joint dimension ") and "nnz_max = 20000000" in reason
    assert f"joint dimension {new.diagnostics.space.dimension} <= joint_dimension_max = 4096" in reason


def test_a_forced_level_reports_the_choice_auto_would_have_made(two_ion) -> None:  # type: ignore[no-untyped-def]
    preset, table = two_ion
    forced = run(
        Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)),
        preset.device,
        20,
        table=table,
        level=FidelityLevel.GATE_LOCAL,
        numerics=FAST,
    )
    assert forced.diagnostics.level == "GATE_LOCAL"
    assert forced.diagnostics.level_reason.startswith(
        "GATE_LOCAL forced by the caller; level='auto' would choose JOINT_EXACT"
    )
