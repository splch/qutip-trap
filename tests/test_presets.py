"""The example devices of ``device/presets.py`` (PLAN.md Section 3.2): the 171Yb+ chain is the validated fixture, hash for
hash, with the drive maps the scheduler needs, and the 40Ca+ optical qubit runs the pipeline on a second species."""

from __future__ import annotations

import pytest

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.schedule import default_gate_drives
from qutip_trap.device.model import BeamRoles
from qutip_trap.device.presets import (
    CA40_DETECTION_WINDOW_S,
    CA40_TRAP_HZ,
    DevicePreset,
    ca40_optical,
    ca40_optical_recipe,
    yb171_chain,
)
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.light.roles import gate_beams
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics
from qutip_trap.prep.recipe import recipe_of, run_preparation, standard_recipe
from qutip_trap.run.job import register_fidelity
from qutip_trap.species import species
from tests.m6_fixtures import circuit_fixture

GPI2 = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
CA40_WINDOWS = (1e-3, 2e-3, 3e-3)


def test_yb171_chain_is_the_validated_fixture_device() -> None:
    for n, waist in ((2, 2.5e-6), (3, 2.0e-6)):
        preset = yb171_chain(n, address_waist_m=waist)
        fixture = circuit_fixture(n, address_waist_m=waist)
        assert isinstance(preset, DevicePreset) and preset.n_ions == n
        assert preset.device.hash() == fixture.device.hash(), (
            "the preset and the validated fixture are one device"
        )
        assert (
            preset.gate_drives == fixture.gate_drives
            and preset.entangling_drives == fixture.entangling_drives
        )
        assert preset.detection_beam == fixture.detection_beam == len(preset.device.beams) - 1
        # the device carries the drive maps as its roles, so the scheduler's default reads them without ambiguity
        assert preset.device.roles == BeamRoles(
            gate=preset.gate_drives, entangling=preset.entangling_drives, detection=preset.detection_beam
        )
        assert default_gate_drives(preset.device) == preset.gate_drives
        assert preset.device.preparation is not None and preset.device.noise.is_quiet(preset.device)


def test_preset_overrides_and_refusals() -> None:
    a = yb171_chain(2)
    b = yb171_chain(2, phase_continuous=True)
    assert a.device.hash() != b.device.hash() and b.device.hardware.phase_continuous
    assert not a.device.hardware.phase_continuous
    with pytest.raises(ValueError):
        yb171_chain(0)
    single = yb171_chain(1)
    assert single.n_ions == 1 and single.gate_drives[0].beams == (2, 3)


# ---- the 40Ca+ optical qubit ------------------------------------------------------------------------------------------------


def test_standard_recipe_refuses_the_optical_qubit_with_a_clear_error() -> None:
    """``standard_recipe`` covers hyperfine qubits whose lower level is an F = 0 state, which is why the 40Ca+ preset carries
    a recipe of its own."""
    device = ca40_optical(1).device
    assert species("40Ca+").qubit == ("S1/2 mJ=-1/2", "D5/2 mJ=-1/2")
    with pytest.raises(NotImplementedError, match="no standard pump for a 40Ca"):
        standard_recipe(device)


def test_the_ca40_preset_derives_a_non_zero_e2_rabi_frequency_and_one_gate_beam() -> None:
    """The Delta m = 0 E2 geometric factor vanishes for the obvious k = y, pol = x choice, so the preset's 729 nm beam runs
    at k = (1, 1, 0)/sqrt 2 with the polarization at 135 degrees and derives 34.7 kHz."""
    from tests.m4_fixtures import derived_seeds

    preset = ca40_optical(1)
    device = preset.device
    assert gate_beams(device) == (0,), "only the 729 nm quadrupole beam is a gate drive"
    assert set(preset.gate_drives) == {0}
    assert preset.gate_drives[0].kind == "optical_E2" and preset.gate_drives[0].beams == (0,)
    assert preset.entangling_drives == {}, "no light-shift pair: two-qubit gates are refused"
    rabi, stark = derived_seeds(device, preset.gate_drives)
    assert rabi[(0, 0)] == pytest.approx(34742.25, rel=1e-4)
    # the differential light shift of the E2 qubit is a few hertz from the resonant detection light, not zero
    assert abs(stark[(0, 0)]) < 10.0
    assert device.detector.window_s == CA40_DETECTION_WINDOW_S
    assert device.trap.omega_hz == CA40_TRAP_HZ


def test_the_ca40_recipe_prepares_the_lower_qubit_level_with_the_doppler_occupations() -> None:
    """Doppler cooling on 397 nm at -Gamma/2 with the 866 nm repump, then the sigma- pump into S1/2 mJ = -1/2; no sideband
    stage, so the Doppler occupations stand: nbar 12.9, 8.9, 9.1 at the 2.0 MHz axial and 3.0/2.9 MHz radial modes, and a
    preparation error of 1.06e-5 (the residual D3/2 population), with the 40Ca+ P1/2 rate read as
    ``conv.ca40_linewidth_reading``."""
    device = ca40_optical(1).device
    recipe = recipe_of(device)
    assert recipe is device.preparation and recipe.sideband is None
    assert recipe.pump_target == ("S1/2 mJ=-1/2",) and recipe.levels == ("S1/2", "P1/2", "D3/2")
    assert any("no sideband-cooling stage" in n for n in recipe.notes)
    assert any("sigma- light along B" in n for n in recipe.notes)
    run_prep = run_preparation(device, recipe)
    assert run_prep.preparation_error(0) == pytest.approx(1.06e-5, rel=5e-2)
    assert sorted(round(v, 1) for v in run_prep.nbar.values()) == [8.9, 9.1, 12.9]
    # the weak pump against the strong repump is what keeps the metastable D3/2 residue out of the error
    hot = ca40_optical_recipe(device, s_pump_397=0.5, s_pump_866=3.0, pump_duration_s=20e-6)
    assert run_preparation(device, hot).preparation_error(0) > 0.1


@pytest.mark.slow
def test_run_completes_one_gpi2_on_the_optical_qubit() -> None:
    """The pipeline end to end on a second species: GPi2(0) on |0> gives the Section 7.6 state, the register infidelity is
    the two frozen radial modes' Debye-Waller loss at the Doppler occupations (5.98e-4), and the shelving readout works."""
    preset = ca40_optical(1)
    result = (
        Machine(preset.device, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-4)))
        .calibrated(
            pairs=list(GPI2.entangling_pairs()), detection_records=600, detection_windows_s=CA40_WINDOWS
        )
        .run(GPI2, 200, keep_final_state=True)
    )
    diagnostics = result.diagnostics
    assert diagnostics.level == "JOINT_EXACT"
    # every mode is frozen or dropped: a carrier pulse displaces nothing, so the register is the whole space
    assert diagnostics.space.dims == [2]
    assert diagnostics.mode_class == {0: "dropped", 1: "frozen", 2: "frozen"}
    assert 1.0 - register_fidelity(result) == pytest.approx(5.98e-4, rel=5e-2)
    # the Fock sum over two Doppler-hot frozen modes needs many branches, and reports what it dropped
    assert diagnostics.branches > 500 and 0.01 < diagnostics.dropped_branch_weight < 0.1
    eps_b, eps_d = result.spam["q0"]
    assert eps_b < 5e-3 and eps_d < 5e-3, (eps_b, eps_d)
    assert result.spam["q0.state_preparation"][0] < 1e-4
    assert abs(result.probabilities.get("0", 0.0) - 0.5) < 5.0 * result.error_bars["0"]
    assert set(result.probabilities) <= {"0", "1"}


def test_run_refuses_a_two_qubit_circuit_on_a_device_with_no_entangling_drive() -> None:
    """No far-detuned 398.5 nm pair, so the Section 4.4.4 light-shift force cannot be played: a circuit with a two-qubit
    gate is refused rather than silently mis-scheduled."""
    preset = ca40_optical(2)
    bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    with pytest.raises(Exception) as excinfo:
        (
            Machine(
                preset.device, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2))
            )
            .calibrated(
                pairs=list(bell.entangling_pairs()), detection_records=200, detection_windows_s=CA40_WINDOWS
            )
            .run(bell, 10)
        )
    assert excinfo.type is not AssertionError
