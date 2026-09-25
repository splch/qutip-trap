"""``run`` on a 40Ca+ optical qubit (M6 audit items D-1 and P1-10/11).

``prep.recipe.standard_recipe`` refuses any qubit whose lower level is not an F = 0 hyperfine level, i.e. every optical and
Zeeman qubit, and before this milestone no ``PreparationRecipe`` for any other species existed anywhere in the repository, so
the whole M6 pipeline was exercised on 171Yb+ alone while Section 3.2's preset list names "171Yb+ chain, 40Ca+ optical,
43Ca+ clock, 137Ba+". ``device.presets.ca40_optical`` is the escape: a complete Device with a hand-written recipe, on which
compile -> calibrate -> schedule -> prepare -> evolve -> readout -> Result runs for a single-qubit circuit.
"""

from __future__ import annotations

import pytest

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.device.presets import (
    CA40_DETECTION_WINDOW_S,
    CA40_TRAP_HZ,
    ca40_optical,
    ca40_optical_recipe,
)
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.light.roles import gate_beams
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics
from qutip_trap.prep.recipe import recipe_of, run_preparation, standard_recipe
from qutip_trap.run.job import register_fidelity
from qutip_trap.species import species

GPI2 = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
WINDOWS = (1e-3, 2e-3, 3e-3)


def test_standard_recipe_still_refuses_the_optical_qubit_with_a_clear_error() -> None:
    """The restriction the preset works around, asserted so it cannot silently widen or narrow: ``standard_recipe`` covers
    hyperfine qubits whose lower level is an F = 0 state, and says so."""
    device = ca40_optical(1).device
    assert species("40Ca+").qubit == ("S1/2 mJ=-1/2", "D5/2 mJ=-1/2")
    with pytest.raises(NotImplementedError, match="no standard pump for a 40Ca"):
        standard_recipe(device)


def test_the_preset_derives_a_non_zero_e2_rabi_frequency_and_one_gate_beam() -> None:
    """The 729 nm beam geometry matters: the Delta m = 0 E2 geometric factor vanishes for the obvious k = y, pol = x choice
    (M4 finding), so the preset uses k = (1, 1, 0)/sqrt 2 with the polarization at 135 degrees and derives 34.7 kHz."""
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


def test_the_recipe_prepares_the_lower_qubit_level_with_the_doppler_occupations() -> None:
    """Doppler cooling on 397 nm at -Gamma/2 with the 866 nm repump, then the sigma- pump into S1/2 mJ = -1/2; no sideband
    stage, so the Doppler occupations stand (nbar 12.9, 8.9, 9.1 at the 2.0 MHz axial and 3.0/2.9 MHz radial modes).

    Both pins moved on 2026-09-08 with the 40Ca+ P1/2 rate (ledger conv.ca40_linewidth_reading): the total rate is now
    Hettrich et al. 2015's measured lifetime, 23.0526 MHz instead of the quoted 21.57 MHz read as a total (+6.9 %), and the
    D3/2 branching is Ramm et al. 2013's 0.06435 instead of Section 8.1's 0.06 (+7.3 %).
      - nbar 15.1/9.6/10.0 -> 12.9/8.9/9.1, i.e. DOWN 14 %, almost all of it from the total rate. It is not the textbook
        nbar_D ~ Gamma/4nu, which would rise: this cycle's magic-angle LINEAR polarization puts it in the coherent
        dark-state regime, where what matters is the Zeeman splitting measured in linewidths, and B = 5 G is fixed. Scaling
        B with Gamma restores the linear scaling (nbar rises 5.1-7.4 % against the expected 6.9 %), which is what isolates
        the fixed field as the cause.
      - preparation error 6.11e-5 -> 1.06e-5, about evenly split (2.81e-5 from the total rate alone, 2.51e-5 from the
        branching alone) but by two different routes. The residual D3/2 population IS the error. The branching acts through
        the 866/397 intensity ratio: at fixed s_pump the 866 nm intensity follows I_sat(866), i.e. the 866 nm PARTIAL rate,
        which rose 14.6 % where the 397 nm one fell 0.5 %. The total rate acts mostly through the FIXED 100 us pump
        duration, worth 6.9 % more pumping in units of 1/Gamma: at 93.57 us the total-rate-only error is 5.17e-5 rather
        than 2.81e-5, so the fixed duration accounts for 72 % of it and the fixed B = 5 G for the rest."""
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
    """The M6 pipeline end to end on a second species: GPi2(0) on |0> gives the Section 7.6 state, the register infidelity is
    the two frozen radial modes' Debye-Waller loss at the Doppler occupations, and the shelving readout works.

    The infidelity fell 6.86e-4 -> 5.98e-4 (-13 %) on 2026-09-08: it is the Debye-Waller loss at nbar, and the 40Ca+ Doppler
    occupations of the two frozen radial modes fell 9.64/10.04 -> 8.88/9.11 with the 397 nm rate (ledger
    conv.ca40_linewidth_reading, anchor.ca40.gamma_dependent_repins). The loss is not exactly linear in (2 nbar + 1), which
    would predict -8 %."""
    preset = ca40_optical(1)
    result = (
        Machine(preset.device, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-4)))
        .calibrated(
            pairs=list(GPI2.entangling_pairs()), **{"detection_records": 600, "detection_windows_s": WINDOWS}
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
    """The preset's honest limit: no far-detuned 398.5 nm pair, so the Section 4.4.4 light-shift force cannot be played and a
    circuit with a two-qubit gate is refused rather than silently mis-scheduled."""
    preset = ca40_optical(2)
    bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    with pytest.raises(Exception) as excinfo:
        (
            Machine(
                preset.device, numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2))
            )
            .calibrated(
                pairs=list(bell.entangling_pairs()),
                **{"detection_records": 200, "detection_windows_s": WINDOWS},
            )
            .run(bell, 10)
        )
    assert excinfo.type is not AssertionError
