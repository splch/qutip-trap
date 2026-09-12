"""The option objects of docs/api_implementation_plan.md 1.4 (``qutip_trap/options.py``): every ``SolverOptions`` field has
exactly one home, the defaults reproduce today's, validation errors are ``SolverOptions``'s, mappings are accepted, and
``to_run_kwargs`` names exactly the keyword arguments of ``run`` that the objects replace."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.options import (
    GateLocal,
    Integration,
    Numerics,
    Parallel,
    Physics,
    Readout,
    Trajectories,
    Truncation,
    to_run_kwargs,
)
from qutip_trap.run.job import run

HOMES: dict[str, tuple[type, str]] = {
    "atol": (Integration, "atol"),
    "rtol": (Integration, "rtol"),
    "nsteps": (Integration, "nsteps"),
    "integrators": (Integration, "integrators"),
    "rotating_frame": (Integration, "rotating_frame"),
    "propagator_cache": (Integration, "propagator_cache"),
    "joint_dimension_max": (Truncation, "joint_dimension_max"),
    "nnz_max": (Truncation, "nnz_max"),
    "mode_dimension_max": (Truncation, "mode_dimension_max"),
    "boundary_population_max": (Truncation, "boundary_population_max"),
    "freeze_chi_max_rad": (Truncation, "freeze_chi_max_rad"),
    "freeze_alpha_max": (Truncation, "freeze_alpha_max"),
    "branch_weight_min": (Truncation, "branch_weight_min"),
    "margin_check": (Truncation, "margin_check"),
    "margin_element_tol": (Truncation, "margin_element_tol"),
    "lindblad_method": (Trajectories, "lindblad_method"),
    "mesolve_dimension_max": (Trajectories, "mesolve_dimension_max"),
    "ntraj": (Trajectories, "ntraj"),
    "improved_sampling": (Trajectories, "improved_sampling"),
    "trajectory_target_tol": (Trajectories, "trajectory_target_tol"),
    "e_ops_for_target_tol": (Trajectories, "e_ops_for_target_tol"),
    "map_accuracy": (GateLocal, "map_accuracy"),
    "crosstalk_threshold": (GateLocal, "crosstalk_threshold"),
    "register_dm_max_qubits": (GateLocal, "register_dm_max_qubits"),
    "register_ensemble": (GateLocal, "register_ensemble"),
    "tomography_isometry": (GateLocal, "tomography_isometry"),
    "tomography_dropped_weight_max": (GateLocal, "tomography_dropped_weight_max"),
    "tomography_tolerance_keyed": (GateLocal, "tomography_tolerance_keyed"),
    "map": (Parallel, "map"),
    "workers": (Parallel, "workers"),
    "convergence_check": (Numerics, "convergence_check"),
    "scattering_channels": (Physics, "scattering"),
    "scattering_recoil": (Physics, "scattering_recoil"),
    "intensity_noise_channels": (Physics, "intensity_noise_channels"),
    "hardware_chain": (Physics, "hardware_chain"),
}
"""Every ``SolverOptions`` field and its one home; a field added to ``SolverOptions`` without a row here fails the table test."""

ON_THE_CALL = {"circuit", "device", "shots", "table", "level", "seed", "keep_final_state", "progress"}
"""The parameters of ``run`` that stay on the call or on the ``Machine`` (``Machine.table``, ``Machine.level``)."""
DEPRECATED_IN_0_3_0 = {"gate_drives", "entangling_drives", "calibrate_kwargs"}
"""Keyword arguments the objects do not carry: the drive maps live on ``Device.roles``, the calibration on ``Machine.calibrated``."""


def test_every_solver_options_field_has_exactly_one_home() -> None:
    names = {f.name for f in dataclasses.fields(SolverOptions)}
    assert set(HOMES) == names, (sorted(set(HOMES) - names), sorted(names - set(HOMES)))
    for name, (cls, attr) in HOMES.items():
        assert attr in {f.name for f in dataclasses.fields(cls)}, (
            f"{name} -> {cls.__name__}.{attr} does not exist"
        )
    homes = [f"{cls.__name__}.{attr}" for cls, attr in HOMES.values()]
    assert len(set(homes)) == len(homes), "two SolverOptions fields share a home"


def test_defaults_reproduce_the_solver_options_defaults() -> None:
    assert Numerics().to_solver_options(Physics()) == SolverOptions()
    assert Numerics.from_solver_options(SolverOptions()) == Numerics()
    assert Physics.from_solver_options(SolverOptions()) == Physics()


def test_non_default_solver_options_round_trip_through_the_objects() -> None:
    opts = SolverOptions(
        atol=1e-11,
        integrators=("vern9",),
        joint_dimension_max=999,
        mode_dimension_max=96,
        branch_weight_min=1e-3,
        lindblad_method="mcsolve",
        ntraj=7,
        map="serial",
        workers=3,
        map_accuracy=1e-4,
        convergence_check=True,
        scattering_channels=True,
        scattering_recoil="vector",
        intensity_noise_channels=False,
        hardware_chain=False,
        rotating_frame=False,
        margin_element_tol=1e-7,
    )
    numerics = Numerics.from_solver_options(opts)
    physics = Physics.from_solver_options(opts, noise=False)
    assert numerics.to_solver_options(physics) == opts
    assert numerics.integration.atol == 1e-11 and numerics.parallel.workers == 3
    assert (
        physics.scattering == "channels" and physics.scattering_recoil == "vector" and physics.noise is False
    )


@pytest.mark.parametrize(
    ("bad", "solver_bad"),
    [
        (lambda: Integration(atol=-1.0), lambda: SolverOptions(atol=-1.0)),
        (lambda: Integration(integrators=("adams",)), lambda: SolverOptions(integrators=("adams",))),
        (lambda: Truncation(joint_dimension_max=1), lambda: SolverOptions(joint_dimension_max=1)),
        (lambda: Truncation(boundary_population_max=2.0), lambda: SolverOptions(boundary_population_max=2.0)),
        (lambda: Trajectories(ntraj=0), lambda: SolverOptions(ntraj=0)),
        (lambda: GateLocal(map_accuracy=1.5), lambda: SolverOptions(map_accuracy=1.5)),
        (lambda: Parallel(workers=0), lambda: SolverOptions(workers=0)),
    ],
)
def test_validation_errors_are_solver_options_errors(bad, solver_bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError) as new:
        bad()
    with pytest.raises(ValueError) as old:
        solver_bad()
    assert str(new.value) == str(old.value)


def test_the_objects_own_rules() -> None:
    with pytest.raises(ValueError, match="internal_levels is at least 2"):
        Physics(internal_levels=1)
    with pytest.raises(ValueError, match="entangler"):
        Physics(entangler="xx")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scattering_recoil"):
        Physics(scattering_recoil="sideways")  # type: ignore[arg-type]  (SolverOptions has no runtime check of it)
    with pytest.raises(ValueError, match="shot_period_s"):
        Physics(shot_period_s=0.0)
    with pytest.raises(ValueError, match="readout mode"):
        Readout(mode="records")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="povm_samples"):
        Readout(povm_samples=0)
    with pytest.raises(ValueError, match="samples is a positive count"):
        Parallel(samples=0)
    with pytest.raises(ValueError, match="not both"):
        Truncation(enr_group=((0, 1), 2), space=Numerics().truncation.space or _space())


def _space():  # type: ignore[no-untyped-def]
    from tests.fixtures import make_space

    return make_space()


def test_mappings_are_accepted_and_unknown_keys_refused() -> None:
    n = Numerics.from_mapping(
        {"integration": {"atol": 1e-11}, "parallel": {"map": "serial"}, "convergence_check": True}
    )
    assert n == Numerics(
        integration=Integration(atol=1e-11), parallel=Parallel(map="serial"), convergence_check=True
    )
    assert Numerics(integration={"atol": 1e-11}) == Numerics(integration=Integration(atol=1e-11))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="has no field \\['atoll'\\]"):
        Integration.from_mapping({"atoll": 1e-11})
    with pytest.raises(TypeError, match="Numerics.integration takes"):
        Numerics(integration=3)  # type: ignore[arg-type]
    assert Numerics.from_mapping(Numerics(convergence_check=True).asdict()) == Numerics(
        convergence_check=True
    )
    assert Physics.from_mapping({"noise": False, "entangler": "zz"}) == Physics(noise=False, entangler="zz")
    assert Readout.from_mapping({"mode": "full"}) == Readout(mode="full")
    assert Physics().asdict()["builder"] is None and Readout().asdict() == {
        "mode": "fast",
        "discriminator": None,
        "povm_samples": 20_000,
    }


def test_to_run_kwargs_names_exactly_the_keyword_arguments_the_objects_replace() -> None:
    params = inspect.signature(run).parameters
    keyword_only = {name for name, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
    produced = set(to_run_kwargs(Physics(), Numerics(), Readout()))
    assert produced <= set(params), sorted(produced - set(params))
    assert produced == keyword_only - ON_THE_CALL - DEPRECATED_IN_0_3_0, sorted(
        produced ^ (keyword_only - ON_THE_CALL - DEPRECATED_IN_0_3_0)
    )
    # the defaults of the objects are the defaults of run, keyword by keyword
    for name, value in to_run_kwargs(Physics(), Numerics(), Readout()).items():
        default = params[name].default
        if name == "options":
            assert value == SolverOptions() and default is None  # run builds SolverOptions() from None
        elif name == "channels":
            assert value == () and default == ()
        else:
            assert value == default, name
