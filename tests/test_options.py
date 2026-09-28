"""The option objects refuse bad fields, and ``Machine.engine`` plays under the physics switches."""

from __future__ import annotations

import pytest

from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Physics, Readout
from tests.fixtures import make_device, make_space


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Numerics(atol=-1.0), "atol is a positive"),
        (lambda: Numerics(nsteps=0), "nsteps is a positive"),
        (lambda: Numerics(integrators=("rk4",)), "the integrators are"),
        (lambda: Numerics(caps={0: 1}), "at least 2 levels"),
        (lambda: Numerics(map="threads"), "map is 'serial', 'parallel' or 'loky'"),  # type: ignore[arg-type]
        (lambda: Numerics(lindblad_method="me"), "lindblad_method"),  # type: ignore[arg-type]
        (lambda: Numerics(integrators=("adams",)), "multistep"),
        (lambda: Numerics(joint_dimension_max=1), "size guards"),
        (lambda: Numerics(boundary_population_max=2.0), "boundary_population_max"),
        (lambda: Numerics(ntraj=0), "ntraj"),
        (lambda: Numerics(map_accuracy=1.5), "map_accuracy"),
        (lambda: Numerics(workers=0), "workers"),
        (lambda: Numerics(samples=0), "samples is a positive count"),
        (lambda: Numerics(enr_group=((0, 1), 2), space=make_space()), "not both"),
        (lambda: Physics(internal_levels=1), "internal_levels is at least 2"),
        (lambda: Physics(entangler="xx"), "entangler"),
        (lambda: Physics(scattering_recoil="sideways"), "scattering_recoil"),
        (lambda: Physics(shot_period_s=0.0), "shot_period_s"),
        (lambda: Readout(mode="records"), "readout mode"),
        (lambda: Readout(povm_samples=0), "povm_samples"),
    ],
)
def test_the_objects_refuse_bad_fields(make, match) -> None:
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Numerics(integrators="dop853"), "a sequence of names"),
        (lambda: Physics(noise="off"), "Physics.noise is True or False"),
        (lambda: Machine(None), "Machine.device is a Device"),
        (lambda: Machine(make_device(), physics=Numerics()), "Machine.physics is a Physics"),
    ],
)
def test_a_wrong_type_is_refused_at_construction(make, match) -> None:
    with pytest.raises(TypeError, match=match):
        make()


def test_the_machine_engine_plays_under_the_physics_switches() -> None:
    physics = Physics(
        noise=False,
        scattering="channels",
        scattering_recoil="vector",
        intensity_noise_channels=False,
        hardware_chain=False,
    )
    eng = Machine(make_device(), physics=physics).engine
    assert not eng.device_channels and eng.scattering_channels and eng.scattering_recoil == "vector"
    assert not eng.intensity_noise_channels and not eng.hardware_chain
